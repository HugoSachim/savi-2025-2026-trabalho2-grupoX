#!/usr/bin/env python3

import os
import signal
import argparse
import torch
from datetime import datetime

# Importa as suas classes (certifique-se que os nomes dos ficheiros estão corretos)
from dataset import CustomMNISTDataset as Dataset  # Ajustei para o nome que criamos
from model import ModelDetectionOptimized           # O modelo CNN para 128x128
from trainer import Trainer

def sigintHandler(signum, frame):
    print('SIGINT received. Exiting gracefully.')
    exit(0)

def main():
    # ------------------------------------
    # Setup argparse
    # ------------------------------------
    parser = argparse.ArgumentParser()

    # Caminho para a pasta que contém 'train' e 'test' (cada uma com 'images' e 'labels')
    parser.add_argument('-df', '--dataset_folder', type=str,
                        default='/home/hogu/Desktop/savi-2025-2026-trabalho2-grupoX/Tarefa_2/data_versao_C/mnist_detection') 
    parser.add_argument('-pe', '--percentage_examples', type=float, default=0.05,
                        help='Percentagem de exemplos a usar')
    parser.add_argument('-ne', '--num_epochs', type=int, default=10,
                        help='Número de épocas')
    parser.add_argument('-bs', '--batch_size', type=int, default=32,
                        help='Batch size (32 é bom para 128x128)')
    parser.add_argument('-ep', '--experiment_path', type=str,
                        default='./experiments',
                        help='Onde salvar os resultados.')
    parser.add_argument('-rt', '--resume_training', action='store_true',
                        help='Retomar treino do último checkpoint.')

    args = vars(parser.parse_args())

    # ------------------------------------
    # Configuração do Experimento
    # ------------------------------------
    signal.signal(signal.SIGINT, sigintHandler)

    # Nome do experimento baseado na data
    experiment_name = datetime.today().strftime('%Y-%m-%d_%H-%M')
    args['experiment_full_name'] = os.path.join(args['experiment_path'], experiment_name)

    os.makedirs(args['experiment_full_name'], exist_ok=True)

    # ------------------------------------
    # Criar Datasets
    # ------------------------------------
    # Importante: O Dataset deve estar a dividir as coordenadas por 128.0!
    train_dataset = Dataset(args, is_train=True)
    test_dataset = Dataset(args, is_train=False)

    # ------------------------------------
    # Criar o Modelo (Otimizado para Detecção)
    # ------------------------------------
    # max_digits: número máximo de números que a rede tentará encontrar por imagem
    model = ModelDetectionOptimized(max_digits=5)

    # Se tiver GPU disponível, movemos o modelo para lá
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ------------------------------------
    # Iniciar o Trainer
    # ------------------------------------
    # O Trainer agora usa MSELoss e desenha caixas vermelhas no draw()
    trainer = Trainer(args, train_dataset, test_dataset, model)

    print(f"Starting training on device: {device}")
    trainer.train()  # Corre o loop de treino e as validações visuais

    # Nota: Removi o trainer.evaluate() porque ele tentava calcular 
    # matriz de confusão de classes, o que não faz sentido para detecção de onde está o número.
    # A avaliação agora é feita visualmente pelas imagens salvas na pasta do experimento.

if __name__ == '__main__':
    main()