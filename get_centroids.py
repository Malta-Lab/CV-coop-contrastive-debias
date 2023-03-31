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
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

from loaders import CelebA, build_dali_loader, get_dali_loaders, UTKface
from models.autoencoder import get_autoencoder
from models.classifier import get_classifier
from losses import ReconstructionContrastiveLoss
from losses import ReconstructionLoss
from utils.general import set_seeds, get_dataset
from utils import define_weight
from utils.metrics import get_metrics_binarybias_cl, get_metrics_racebias_cl

def get_args():
    parser = argparse.ArgumentParser(description='Parameters for AE train')

    parser.add_argument('--classifier',     default='resnet', choices=['simple', 'unet', 'resnet'], type=str)
    parser.add_argument('--feat_layer',     default='preconv', choices=['img', 'preconv', 'layer1', 'layer2', 'layer3', 'layer4', 'all'], type=str)
    parser.add_argument('--batch_size',     default=128, type=int, help='Size for each mini batch.')
    parser.add_argument('--dataset',        default='celeba', choices=['mnist', 'celeba', 'utkface'], type=str)
    parser.add_argument('--dataset_path',   default='./dataset/', type=str)
    parser.add_argument('--transform',      default='default', choices=['default', '220-180'], type=str, help='Transform size for the images')
    parser.add_argument('--gpu',            default='-1', help='GPU number')
    parser.add_argument('--load_path_cl',   default='./checkpoints/celeba/', type=str, help='Path to load the classifier.')
    parser.add_argument('--name',           default='test', help='Run name on Tensorboard.')
    parser.add_argument('--target',         default='gender', type=str, help='Target attribute')
    parser.add_argument('--bias',           default='race', type=str, help='Bias attribute')
    parser.add_argument('--is_biased',      default='False', type=str, help='If the train dataset is biased or not')
    parser.add_argument('--bias_prop',      default=0.1, type=float, help='Proportion of biased data')
    parser.add_argument('--seed',           default=42, type=int, help='Seed for random number generators')

    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    return args

def centroid_pass(
    classifier: nn.Module,
    feat_layer: str,
    mini_batch: List[torch.Tensor]
) -> List[torch.Tensor]:
    
    images, labels, bias = mini_batch
    classifier.eval()
    classifier.requires_grad_(False)

    #TODO precisamos passar por todo o modelo ou apenas até a camada desejada?
    with torch.no_grad():
        features, pred = classifier(x=images, output_features=True, layer=feat_layer)

        #verify if the prediction is correct and create lists of 0 and 1 predictions
        sum_pred_label = torch.argmax(pred, dim=1) + labels
        
        correct_0 = sum_pred_label == 0
        correct_1 = sum_pred_label == 2
        
        #get features from idx fom correct_0 and correct_1
        features_0_batch = features[correct_0]
        features_1_batch = features[correct_1]

        features_0_batch = torch.tensor(features_0_batch, device = 'cpu')
        features_1_batch = torch.tensor(features_1_batch, device = 'cpu')
            
        # features_0_batch = features_0_batch.detach()
        # features_1_batch = features_1_batch.detach()

        # features_0_batch = features_0_batch.cpu()
        # features_1_batch = features_1_batch.cpu()

    return features_0_batch, features_1_batch


if __name__ == "__main__":
    args = get_args()
    print(f'Starting script. Name: {args.name}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # load dataset
    print('Loading Dataset:', args.dataset)
    print('Dataset path:', args.dataset_path)
    nvidia_train = get_dataset(args, split='train', is_biased=args.is_biased, bias_prop=args.bias_prop)
    nvidia_valid = get_dataset(args, split='valid', is_biased=args.is_biased, bias_prop=args.bias_prop)

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

    # Classifier
    print('Loading Models')
    classifier = get_classifier(args.classifier)
    classifier = classifier(2).to(device)
    print(f'Loading cl weights from: {args.load_path_cl}')
    weights_cl = torch.load(
        f'{args.load_path_cl}',
        map_location=device
    )
    classifier.load_state_dict(weights_cl)
    classifier.eval()
    classifier.requires_grad_(False)

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

    #create tensors to store centroids
    centroid_0 = []
    centroid_1 = []

    for idx, data in enumerate(tqdm(train_data, ncols=120, desc='Getting centroids')):
        image, bias_labels = data[0]["image"], data[0]["bias_and_labels"]
        
        #getting bias and labels unmapping them
        bias_labels = bias_labels.squeeze()
        labels = bias_labels.clone()
        bias = bias_labels.clone()
        for i in range(bias_labels.size(0)):
            labels[i], bias[i] = nvidia_train.unmap(bias_labels[i].item())
        image = transform_batch(image)
        labels = labels.type(torch.LongTensor).to(device)
        bias = bias.to(device)
        
        #getting features from centroid pass
        features_0_batch, features_1_batch = centroid_pass(
            classifier, 
            args.feat_layer,
            [image, labels, bias]
        )

        #appending each feature batch to centroid list
        centroid_0.append(features_0_batch)
        centroid_1.append(features_1_batch)
        # centroid_0=torch.cat((centroid_0, features_0_batch), 0)
        # centroid_1=torch.cat((centroid_1, features_1_batch), 0)

        # if idx == 50:
        #     break

    #split centroids list into smaller lists
    print('Splitting centroids...')
    number_of_chunks = 10

    chunk_size_0 = len(centroid_0)//number_of_chunks
    chunk_size_1 = len(centroid_1)//number_of_chunks

    # cent_0=torch.Tensor(size=(0,lenght_0//number_of_chunks))
    # cent_1=torch.Tensor(size=(0,lenght_1//number_of_chunks))

    cent_0=[]
    cent_1=[]



    for i in range(number_of_chunks):
        # rnd_0=np.random.randint(0, centroid_0.size(0), chunk_size_0)
        # rnd_1=np.random.randint(0, centroid_1.size(0), chunk_size_1)
        
        #getting random indexes
        rnd_0=np.random.randint(0, len(centroid_0), chunk_size_0)
        rnd_1=np.random.randint(0, len(centroid_1), chunk_size_1)
    
        #getting random chunks
        chunk_list_0 = [centroid_0[i] for i in rnd_0]
        chunk_list_1 = [centroid_1[i] for i in rnd_1]

        #concatenating chunks
        chunk_0 = torch.cat(chunk_list_0, dim=0)
        chunk_1 = torch.cat(chunk_list_1, dim=0)
        
        #get mean of chunks - generating centroids
        chunk_0 = chunk_0.mean(dim=0)
        chunk_1 = chunk_1.mean(dim=0)

        # print('premean--->', chunk_0.size())
        # print('premean--->', chunk_1.size())

        # print('postmean--->', chunk_0.size())
        # print('postmean--->', chunk_1.size())

        cent_0.append(chunk_0)
        cent_1.append(chunk_1)
        
        # cent_0=torch.cat((cent_0, chunk_0[None]), 0)
        # cent_1=torch.cat((cent_1, chunk_1[None]), 0)

    print(len(cent_0))
    print(len(cent_1))

    print(cent_0[0].size())
    print(cent_1[0].size())

    #saving centroid features
    print('Saving centroids...')
    model_path = os.path.join(f'./centroid_{args.dataset}/{args.name}')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    c0_path = os.path.join(model_path, f'centroid_0{args.feat_layer}.pt')
    c1_path = os.path.join(model_path, f'centroid_1{args.feat_layer}.pt')

    torch.save(cent_0, c0_path)
    torch.save(cent_1, c1_path)