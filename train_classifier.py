import argparse
from collections import defaultdict
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tensorboard_wrapper.tensorboard import Tensorboard
import torch
from torch import nn, optim
from torchvision import transforms
from tqdm import tqdm
from utils.metrics import get_metrics_binarybias_cl, get_metrics_racebias_cl
from loaders import CelebA, build_dali_loader, get_dali_loaders, UTKface
from models.classifier import get_classifier
from utils.general import set_seeds, get_dataset


def get_args():
    parser = argparse.ArgumentParser(description='Parameters for classifier train')

    parser.add_argument('--classifier',     default='resnet', choices=['resnet', 'simple', 'unet'], type=str)
    parser.add_argument('--batch_size',     default=128, type=int, help='Size for each mini batch.')
    parser.add_argument('--dataset',        default='celeba', choices=['mnist', 'celeba', 'utkface'], type=str)
    parser.add_argument('--dataset_path',   default='./datasets/', type=str)
    parser.add_argument('--transform',      default='default', choices=['default', '220-180'], type=str, help='Transform size for the images')
    parser.add_argument('--lr',             default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--epoch',          default=1000, type=int)
    parser.add_argument('--early_stop',     default=15, type=int)   
    parser.add_argument('--gpu',            default='-1', help='GPU number')
    parser.add_argument('--load_path',      default=None, type=str, help='Path to load the model.')
    parser.add_argument('--name',           default='test_utkface', help='Run name on Tensorboard.') 
    parser.add_argument('--target',         default='gender', type=str, help='Target attribute')
    parser.add_argument('--bias',           default='race', type=str, help='Bias attribute')
    parser.add_argument('--seed',           default=42, type=int, help='Seed for the experiment')
    parser.add_argument('--is_biased',      default='False', type=str, help='If the dataset is biased or not')
    parser.add_argument('--bias_prop',      default=0.1, type=float, help='Proportion of biased data')
 
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    return args
    
def iter(
    classifier: nn.Module, 
    mini_batch: List[torch.Tensor],
    optimizer: nn.Module,
    criterion: nn.Module,
    train: bool
) -> Tuple[torch.Tensor]:

    optimizer.zero_grad()
    if train:
        classifier.train()
        classifier.requires_grad_(True)
    
    else:
        classifier.eval()
        classifier.requires_grad_(False)

    images, labels, bias = mini_batch

    #forward
    _, pred = classifier(images)

    #getting loss
    loss = criterion(pred, labels)

    #backpropagation
    if train:
        loss.backward()
        optimizer.step()
    
    #predictions
    predictions = torch.argmax(pred, dim=1)

    return loss, predictions, bias

if __name__ == "__main__":
    args = get_args()
    print(f'Starting script. Name: {args.name}')
    set_seeds(args.seed)

    board = Tensorboard(args.name, f'./runs_paper/{args.name}', delete=True)  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    # load dataset
    print('Loading Dataset:', args.dataset)
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

    # Models
    print("Loading models")
    classifier = get_classifier(classifier=args.classifier)
    #classifier = classifier(img_channels=3, num_layers=18, num_classes=2).to(device)
    classifier = classifier(2).to(device)
    classifier.train()
    classifier.requires_grad_(True)

    torch.backends.cudnn.benchmark = True

    # Optimizers and criterions
    print ('Defining optimizer and criterions')
    print(f'Using Learning Rate: {args.lr}')
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training utils
    print(f'Using early stop: {args.early_stop}')
    early_stop = 0
    best_acc = 0
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

    for epoch in range(args.epoch):
        if early_stop == args.early_stop:
            print('Early stop reached. Stopping training...')
            break
            
        for is_train, description in zip([True, False], ["Train", "Valid"]):
            epoch_metrics = defaultdict(list)
            epoch_pred = []
            epoch_bias = []
            epoch_loss = []
            epoch_labels = []

            data_loader = train_data if is_train else valid_data
            
            pbar = tqdm(data_loader, ncols=120)
            pbar.set_description_str(desc=f'{description} {epoch}', refresh=True)

            for idx, data in enumerate(pbar):
                image, bias_labels = data[0]["image"], data[0]["bias_and_labels"]
                bias_labels = bias_labels.squeeze()

                labels = bias_labels.clone()
                bias = bias_labels.clone()

                #TODO pode ser nvidia_train aqui??
                for i in range(bias_labels.size(0)):
                    labels[i], bias[i] = nvidia_train.unmap(bias_labels[i].item())

                image = transform_batch(image)
                
                #TODO esse .type deveria estar aqui???
                labels = labels.type(torch.LongTensor).to(device)
                bias = bias.to(device)

                if idx == 0:   #saving autoencoder input and output images from first batch
                    board.add_grid(prior=description, input=image)

                loss, prediction, bias = iter(classifier, [image, labels, bias], optimizer, criterion, train = is_train)
                
                epoch_pred.extend(prediction.tolist())
                epoch_bias.extend(bias.tolist())
                epoch_loss.append(loss.item())
                epoch_labels.extend(labels.tolist())
                
                pbar.set_postfix_str(
                    s=f'Mean loss: {np.mean(epoch_loss).mean():.4f}',
                    refresh=True
                )

            #getting metrics from epoch_pred and epoch_bias
            if args.dataset == 'celeba':
                epoch_metrics = get_metrics_binarybias_cl(epoch_labels, epoch_pred, epoch_bias)
            if args.dataset == 'utkface':
                epoch_metrics = get_metrics_racebias_cl(epoch_pred, epoch_bias)
            epoch_metrics['Loss'] = np.mean(epoch_loss)
            board.add_scalars(prior=description, **epoch_metrics) 

        if best_acc < epoch_metrics['Acc']:
        #if epoch == args.epoch - 1:
            print('Best epoch. Valid acc:', epoch_metrics['Acc'], 'Saving model...')  
            print(epoch_metrics)
            early_stop = 0
            best_acc = epoch_metrics['Acc']                      
            model_path = os.path.join('./checkpoints', 'classifier', args.name)
            if not os.path.exists(model_path):
                os.makedirs(model_path)

            torch.save(classifier.state_dict(), f'{model_path}/cl_best_model.ckpt')

        else:
        #if epoch == args.epoch - 1:
            early_stop += 1

        board.step()
    
#-------------------------------------------------------------------------------

    #evaluation on unbiased data
    print('Training finished. Loading best model...')
    classifier = []
    classifier = get_classifier(args.classifier)
    classifier = classifier(2, False).to(device)
    print(f'Loading cl weights from: {model_path}/cl_best_model.ckpt')
    weights_cl = torch.load(
        f'{model_path}/cl_best_model.ckpt',
        map_location=device
    )
    classifier.load_state_dict(weights_cl)
    classifier.eval()
    classifier.requires_grad_(False)

    print('Loading unbiased data and pipelines...')
    if args.dataset == 'celeba':
        file_root= os.path.join(args.dataset_path, 'celeba', 'img_align_celeba')
    
    nvidia_valid_unbiased = get_dataset(args, split='valid', is_biased='equal_splits', bias_prop=0.0)

    valid_unbiased_pipe, _ = get_dali_loaders(args.batch_size)
    valid_unbiased_pipe = valid_unbiased_pipe(file_root=file_root, split=nvidia_valid_unbiased)
    unbiased_valid_data = build_dali_loader(valid_unbiased_pipe, len(nvidia_valid_unbiased))
    
    is_train=False
    description = "Valid_unbiased"
    epoch_metrics = defaultdict(list)
    epoch_pred = []
    epoch_bias = []
    epoch_loss = []
    epoch_labels = []

    data_loader = unbiased_valid_data

    pbar = tqdm(data_loader, ncols=120)
    pbar.set_description_str(desc=f'{description}', refresh=True)

    for idx, data in enumerate(pbar):
        image, bias_labels = data[0]["image"], data[0]["bias_and_labels"]
        bias_labels = bias_labels.squeeze()

        labels = bias_labels.clone()
        bias = bias_labels.clone()

        #TODO pode ser nvidia_train aqui??
        for i in range(bias_labels.size(0)):
            labels[i], bias[i] = nvidia_train.unmap(bias_labels[i].item())

        image = transform_batch(image)
        
        #TODO esse .type deveria estar aqui???
        labels = labels.type(torch.LongTensor).to(device)
        bias = bias.to(device)

        if idx == 0:   #saving autoencoder input and output images from first batch
            board.add_grid(prior=description, input=image)

        loss, prediction, bias = iter(classifier, [image, labels, bias], optimizer, criterion, train = is_train)
        
        epoch_pred.extend(prediction.tolist())
        epoch_bias.extend(bias.tolist())
        epoch_loss.append(loss.item())
        epoch_labels.extend(labels.tolist())
        
        pbar.set_postfix_str(
            s=f'Mean loss: {np.mean(epoch_loss).mean():.4f}',
            refresh=True
        )

    #getting metrics from epoch_pred and epoch_bias
    if args.dataset == 'celeba':
        epoch_metrics = get_metrics_binarybias_cl(epoch_labels, epoch_pred, epoch_bias)
    if args.dataset == 'utkface':
        epoch_metrics = get_metrics_racebias_cl(epoch_pred, epoch_bias)
    epoch_metrics['Loss'] = np.mean(epoch_loss)
    board.add_scalars(prior=description, **epoch_metrics) 
    print(epoch_metrics)

    board.close()