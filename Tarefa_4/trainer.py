import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import wandb
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

# Função Collate para lidar com número variável de dígitos por imagem
def detection_collate(batch):
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch] # Mantém labels como lista de tensores
    return images, labels

class Trainer():
    def __init__(self, args, train_dataset, test_dataset, model):
        self.args = args
        self.model = model

        # 1. Ajuste do DataLoader com collate_fn customizado
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=args['batch_size'],
            shuffle=True, collate_fn=detection_collate)
        
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=args['batch_size'],
            shuffle=False, collate_fn=detection_collate)

        # 2. Loss de Regressão (MSE) em vez de CrossEntropy
        self.loss_func = nn.MSELoss()

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.001)

        wandb.init(project="mnist_detection", name=self.args['experiment_full_name'], config=self.args)

        if self.args['resume_training']:
            self.loadTrain()
        else:
            self.train_epoch_losses = []
            self.test_epoch_losses = []
            self.epoch_idx = 0

    def prepare_labels(self, labels_list):
        """
        Padroniza as labels para o tamanho fixo que o modelo espera (max_digits, 4).
        Ignoramos a classe (dígito) aqui pois o foco é localização.
        """
        max_d = self.model.max_digits
        padded_labels = []
        
        for l in labels_list:
            # l é [N_digitos, 5] -> 5 colunas: [digito, x, y, w, h]
            # Pegamos apenas as coordenadas (colunas 1 a 4)
            coords = l[:, 1:5]
            
            # Criamos um tensor vazio de zeros [max_digits, 4]
            target = torch.zeros((max_d, 4))
            
            # Preenchemos com os dígitos existentes (limitado ao máximo do modelo)
            num_to_copy = min(coords.shape[0], max_d)
            target[:num_to_copy, :] = coords[:num_to_copy, :]
            
            padded_labels.append(target)
            
        return torch.stack(padded_labels)

    def train(self):
        print(f'Training started. Max epochs = {self.args["num_epochs"]}')

        for i in range(self.epoch_idx, self.args['num_epochs'] + 1):
            self.epoch_idx = i
            self.model.train()
            train_batch_losses = []

            for image_tensor, label_gt_list in tqdm(self.train_dataloader, desc=f"Epoch {i} [Train]"):
                
                # Prepara os alvos (coordenadas x,y,w,h)
                target_tensor = self.prepare_labels(label_gt_list)

                # Forward
                # Saída do modelo: [batch, max_digits, 4]
                pred_tensor = self.model(image_tensor)

                # Loss de regressão entre coordenadas preditas e reais
                batch_loss = self.loss_func(pred_tensor, target_tensor)
                
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                train_batch_losses.append(batch_loss.item())

            # Teste
            self.model.eval()
            test_batch_losses = []
            with torch.no_grad():
                for image_tensor, label_gt_list in self.test_dataloader:
                    target_tensor = self.prepare_labels(label_gt_list)
                    pred_tensor = self.model(image_tensor)
                    batch_loss = self.loss_func(pred_tensor, target_tensor)
                    test_batch_losses.append(batch_loss.item())

            # Logs e Save
            train_loss = np.mean(train_batch_losses)
            test_loss = np.mean(test_batch_losses)
            self.train_epoch_losses.append(train_loss)
            self.test_epoch_losses.append(test_loss)

            wandb.log({"train_loss": train_loss, "test_loss": test_loss, "epoch": i})
            print(f"Epoch {i} - Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
            
            self.saveTrain()
            if i % 5 == 0: self.draw()

    def saveTrain(self):
        checkpoint = {
            'epoch_idx': self.epoch_idx,
            'train_epoch_losses': self.train_epoch_losses,
            'test_epoch_losses': self.test_epoch_losses,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        os.makedirs(self.args['experiment_full_name'], exist_ok=True)
        path = os.path.join(self.args['experiment_full_name'], 'checkpoint.pkl')
        torch.save(checkpoint, path)

    def draw(self):
        # --- 1. Gráfico de Loss (mesma lógica de antes) ---
        plt.figure(1)
        plt.clf()
        plt.plot(self.train_epoch_losses, 'r-', label='Train')
        plt.plot(self.test_epoch_losses, 'b-', label='Test')
        plt.title("Detection Loss (MSE)")
        plt.legend()
        plt.savefig(os.path.join(self.args['experiment_full_name'], 'training_loss.png'))

        # --- 2. Desenhar Predições (Visualização) ---
        self.model.eval()
        with torch.no_grad():
            # Pega apenas um batch do test_dataloader
            images, labels = next(iter(self.test_dataloader))
            
            # Vamos usar apenas a primeira imagem do batch
            img_tensor = images[0:1] # Shape [1, 1, 128, 128]
            preds = self.model(img_tensor) # Shape [1, max_digits, 4]
            
            # Converter o tensor da imagem de volta para PIL (0-1 para 0-255)
            # .squeeze() remove dimensões extras
            img_np = (img_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np, mode='L').convert('RGB')
            draw = ImageDraw.Draw(img_pil)

            # Pegar as predições da primeira imagem [max_digits, 4]
            # As coordenadas estão em 0-1, precisamos desnormalizar (voltar para 0-128)
            image_preds = preds[0] 

            for box in image_preds:
                x_norm, y_norm, w_norm, h_norm = box.cpu().numpy()
                
                # Se a rede prever valores muito pequenos (caixa vazia), ignoramos
                if w_norm < 0.05 or h_norm < 0.05:
                    continue

                # Converter de normalizado (0-1) para pixels (0-128)
                # Assumindo formato YOLO: x, y são o centro
                x_center = x_norm * 128
                y_center = y_norm * 128
                width = w_norm * 128
                height = h_norm * 128

                # Calcular cantos (Top-Left e Bottom-Right) para o ImageDraw
                x0 = x_center - (width / 2)
                y0 = y_center - (height / 2)
                x1 = x_center + (width / 2)
                y1 = y_center + (height / 2)

                # Desenha o retângulo vermelho
                draw.rectangle([x0, y0, x1, y1], outline="red", width=2)

            # Salvar a imagem com os quadrados desenhados
            pred_img_path = os.path.join(self.args['experiment_full_name'], f'val_prediction_epoch_{self.epoch_idx}.png')
            img_pil.save(pred_img_path)
            
            # Opcional: Logar imagem no WandB para acompanhar a evolução visual
            wandb.log({"visual_prediction": wandb.Image(img_pil), "epoch": self.epoch_idx})

        print(f"Grafico e visualização salvos para a época {self.epoch_idx}")