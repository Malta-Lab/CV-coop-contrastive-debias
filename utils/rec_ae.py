import argparse
from argparse import Namespace
from collections import defaultdict
import os
import warnings
from typing import Dict, List

import numpy as np
from tensorboard_wrapper.tensorboard import Tensorboard
import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from loaders import CelebA, build_dali_loader, get_dali_loaders, UTKface
from models.autoencoder import get_autoencoder
from losses import ReconstructionLoss
from utils.general import set_seeds, get_dataset

def iter(
    ae: nn.Module, 
    input: torch.Tensor,
    optimizer: nn.Module, 
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    train: bool
) -> Dict[str, float]:
    
    optimizer.zero_grad()
    if train:
        ae.train()
        ae.requires_grad_(True)
             
    else:
        ae.eval()
        ae.requires_grad_(False)
    
    output = ae(input)

    loss = criterion(input, output)

    #backpropagation
    if train:
        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return loss, output

#if __name__ == "__main__":
def rec_ae_train(args):

    print(f'Starting reconstruction autoencoder pre-train. Name: {args.name}')
    set_seeds(args.seed)

    board = Tensorboard(args.name, f'./runs_rec_ae/{args.name}', delete=True)  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # load dataset
    print('Loading Dataset')
    nvidia_train = get_dataset(args, split='train', is_biased = args.is_biased, bias_prop = args.bias_prop)
    nvidia_valid = get_dataset(args, split='valid', is_biased = args.is_biased, bias_prop = args.bias_prop)

    # define pipelines
    if args.dataset == 'celeba':
        file_root= os.path.join(args.dataset_path, 'celeba', 'img_align_celeba')
    if args.dataset == 'utkface':
        file_root= os.path.join(args.dataset_path, 'UTKface', 'data')
    train_pipe, valid_pipe = get_dali_loaders(args.batch_size)
    train_pipe = train_pipe(file_root=file_root, split=nvidia_train)
    valid_pipe = valid_pipe(file_root=file_root, split=nvidia_valid)
    train_data = build_dali_loader(train_pipe, len(nvidia_train))
    valid_data = build_dali_loader(valid_pipe, len(nvidia_valid))

    torch.backends.cudnn.benchmark = True

    # Models
    print('Loading Models')
    
    autoencoder = get_autoencoder(args.autoencoder)
    autoencoder = autoencoder().to(device)

    # Optimizers and Criterions
    print('Defining optimizer and criterions')
    print(f'Using learning rate: {args.lr}')
    optimizer = optim.Adam(autoencoder.parameters(), lr = args.lr)
    #optimizer = optim.SGD(autoencoder.parameters(), lr=args.lr, momentum=0.9)
    criterion = ReconstructionLoss(nn.MSELoss())
    scaler = torch.cuda.amp.GradScaler()

    # Training utils
    print(f'Using early stop: {args.early_stop}')
    early_stop = 0
    best_loss = np.inf

    print('Using batch size:', args.batch_size)
    
    if args.dataset == 'celeba':
        transform_batch = transforms.Compose([
            lambda x: x.permute(0,3,1,2),
            lambda x: x/255,
            lambda x: x.type(torch.float32),
            transforms.Resize((224, 224)),
        ])   
    #keep 200x200 for utkface
    if args.dataset == 'utkface':
        transform_batch = transforms.Compose([
            lambda x: x.permute(0,3,1,2),
            lambda x: x/255,
            lambda x: x.type(torch.float32),
        ])
    #old transform for celeba
    if args.transform == '220-180':
        print('Using transform 220-180')
        transform_batch = transforms.Compose([
            lambda x: x.permute(0,3,1,2),
            lambda x: x/255,
            lambda x: x.type(torch.float32),
            transforms.Pad(padding=1),
        ])    

    for epoch in range(args.epoch*5):
        for is_train, description in zip([True, False], ["Train", "Valid"]):
            epoch_loss = []

            data_loader = train_data if is_train else valid_data
            
            pbar = tqdm(data_loader, ncols=120)
            pbar.set_description_str(desc=f'{description} {epoch}', refresh=True)

            for idx, data in enumerate(pbar):
                image, _ = data[0]["image"], data[0]["bias_and_labels"]
                image = transform_batch(image)

                loss, output = iter(
                    autoencoder,
                    image,
                    optimizer,
                    criterion,
                    scaler,
                    train=is_train
                ) 

                if idx == 0:   #saving autoencoder input and output images from first batch
                    board.add_grid(prior=description, ae_input=image)
                    board.add_grid(prior=description, ae_output=output)
                
                epoch_loss.append(loss.item())
                         
                pbar.set_postfix_str(
                    s=f'Mean loss: {np.mean(epoch_loss).mean():.4f}',
                    refresh=True
                )
           
            #adding metrics to tensorboard
            epoch_loss = np.mean(epoch_loss)
            board.add_scalar('loss', epoch_loss, prior=description)

        #a partir daqui só se aplica para validação  
        if best_loss > epoch_loss:
            print('Best epoch. Valid loss:', epoch_loss, 'Saving model...')  
            early_stop = 0
            best_loss = epoch_loss                      
            model_path = './checkpoints/rec_ae/'
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            rec_ae_path = f'{model_path}{args.name}.ckpt'

            torch.save(autoencoder.state_dict(), rec_ae_path)

        else:
            early_stop += 1
            if early_stop == args.early_stop:
                print('Early stop reached. Stopping training...')
                board.close()
                return rec_ae_path

        board.step()
    board.close()
    return rec_ae_path
