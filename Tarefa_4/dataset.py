import glob
import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class MNISTDetectionDataset(Dataset):
    def __init__(self, args, is_train):
        self.args = args
        self.train = is_train

        split_name = 'train' if is_train else 'test'
        self.image_path = os.path.join(args['dataset_folder'], split_name, 'images')
        self.label_path = os.path.join(args['dataset_folder'], split_name, 'labels')

        # Busca imagens .jpg ou .png
        self.image_filenames = sorted(glob.glob(os.path.join(self.image_path, "*.jpg")))
        if not self.image_filenames:
            self.image_filenames = sorted(glob.glob(os.path.join(self.image_path, "*.png")))

        num_examples = round(len(self.image_filenames) * args['percentage_examples'])
        self.image_filenames = self.image_filenames[:num_examples]

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # 1. Carregar Imagem
        img_path = self.image_filenames[idx]
        image = Image.open(img_path).convert('L')
        image_tensor = self.transform(image)

        # 2. Carregar Labels (Dígito + Coordenadas)
        basename = os.path.basename(img_path)
        label_filename = os.path.splitext(basename)[0] + ".txt"
        label_file_path = os.path.join(self.label_path, label_filename)

        targets = []
        # Dentro do __getitem__ do seu Dataset
        if os.path.exists(label_file_path):
            with open(label_file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        digit = float(parts[0])
                        
                        # --- NORMALIZAÇÃO AQUI ---
                        # Dividimos os valores em pixels por 128 para ficarem entre 0 e 1
                        x = float(parts[1]) / 128.0
                        y = float(parts[2]) / 128.0
                        w = float(parts[3]) / 128.0
                        h = float(parts[4]) / 128.0
                        
                        targets.append([digit, x, y, w, h])
        
        # Converte para tensor: cada linha é [dígito, x, y, w, h]
        if len(targets) > 0:
            label_tensor = torch.tensor(targets, dtype=torch.float)
        else:
            # Caso não haja dígitos na imagem, retorna tensor vazio com 5 colunas
            label_tensor = torch.zeros((0, 5), dtype=torch.float)

        return image_tensor, label_tensor

# --- FUNÇÃO COLLATE ---
def detection_collate(batch):
    """
    Agrupa imagens em um tensor e labels em uma lista, 
    já que cada imagem pode ter um número diferente de dígitos.
    """
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Empilha as imagens (todas têm o mesmo tamanho 28x28 ou similar)
    images = torch.stack(images, dim=0)
    
    # Retorna as labels como uma LISTA de tensores
    return images, labels

# --- COMO USAR ---
# args = {'dataset_folder': './meu_mnist', 'percentage_examples': 1.0}
# dataset = MNISTDetectionDataset(args, is_train=True)
# loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=detection_collate)