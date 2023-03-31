#evaluating autoencoder on test data

import argparse
from argparse import Namespace
from collections import defaultdict
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
from tensorboard_wrapper.tensorboard import Tensorboard
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

from torch.utils.data import Dataset
from loaders import CelebA, UTKface, build_dali_loader, get_dali_loaders
from models import get_classifier, Resnet
from models.autoencoder import ConvWithConn, AutoEncoder
from losses import ReconstructionContrastiveLoss
from losses import ReconstructionLoss

from utils.metrics import get_metrics_genderbias_cl, get_metrics_racebias_cl
from utils.general import set_seeds
from utils import define_weight

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator

def get_args():
    parser = argparse.ArgumentParser(description='Parameters for AE evaluation')
        
    parser.add_argument('--autoencoder',        default='unet',  choices=['lae', 'cae', 'unet'], type=str)
    parser.add_argument('--batch_size',         default=128, type=int, help='Size for each mini batch.')
    parser.add_argument('--dataset_path',       default='./dataset/', type=str)
    parser.add_argument('--gpu',                default='0', help='GPU number')
    parser.add_argument('--classifier_path',    default='./checkpoints/celeba/best_model.ckpt', type=str, help='Path to load the classifier.')
    parser.add_argument('--autoencoder_path',   default='./checkpoints/celeba/', type=str, help='Path to load the autoencoder.')
    parser.add_argument('--name',               default='test', help='Run name on Tensorboard.')
    parser.add_argument('--temperature',        default=0.5, type=float, help='Temperature for the Contrastive loss')
    parser.add_argument('--dataset',            default='celeba', choices=['mnist', 'celeba'], type=str)
    parser.add_argument('--alpha',             default=1, type=float, help='Weight for the class loss')
    parser.add_argument('--theta',             default=1, type=float, help='Weight for the feature loss')
    parser.add_argument('--recon',             default=0, type=float, help='Weight for the reconstruction loss')
    parser.add_argument('--dyn',               default=True, help='Use dynamic contrastive loss')
    parser.add_argument('--bias',           default='race', choices=['age','gender', 'race'], help='Bias choice for utkface: age, gender or race')
    parser.add_argument('--label',          default='gender', choices=['age','gender', 'race'], help='target for utkface: age, gender or race')
    parser.add_argument('--seed',           default=1, type=int, help='Seed for the experiment')

    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    return args

def get_dataset(args, split:str) -> Dataset:
    if args.dataset == 'celeba':
            return CelebA(
            root = args.dataset_path,
            split= split,
            class_subset = ["Arched_Eyebrows"],
            download = True
        )
    
    if args.dataset == 'utkface':
        return UTKface(
            root = args.dataset_path,
            split = split,
            bias = args.bias,
            label = args.label
        )


def iter(
    ae: AutoEncoder, 
    classifier: Resnet,
    mini_batch: List[torch.Tensor], 
    criterions: List[nn.Module],
    args: Namespace,
) -> Dict[str, float]:

    images, labels, bias = mini_batch

    with torch.no_grad:
        unbiased = ae(images)

        #forward and getting features from the unbiased input classifier
        features, pred = classifier(unbiased, output_features=True, layer='preconv')

        #getting features from the biased input classifier
        gt_features, _ = classifier(images, output_features=True, layer='preconv')

        #getting criterions and calculating losses
        criterion_class, criterion_features, criterion_rec = criterions 
        loss_class = criterion_class(pred, labels)
        loss_features = criterion_features(features, gt_features, pred, labels)
        loss_rec = criterion_rec(unbiased, images)

        # setting loss_rec to 0 when it reaches the specified epoch
        recon = args.recon
        
        loss = (args.alpha * loss_class) + (args.theta * loss_features) + (recon * loss_rec)
        losses = [loss, loss_features, loss_rec, loss_class]

        #correct predictions
        correct = torch.argmax(pred, dim=1) == labels

        return unbiased, losses, correct, bias

if __name__ == '__main__':
    args = get_args()
    print(f'Starting evaluation. Name: {args.name}')
    set_seeds(args.seed)

    board = Tensorboard(args.name, f'./runs/{args.name}', delete=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    # load dataset
    nvidia_test = get_dataset(args, split='test')

    # define pipelines
    if args.dataset == 'celeba':
        file_root= os.path.join(args.dataset_path, 'celeba', 'img_align_celeba')
    if args.dataset == 'utkface':
        file_root= os.path.join(args.dataset_path, 'UTKface', 'data')
    test_pipe = get_dali_loaders(args.batch_size)
    test_pipe = test_pipe(file_root=file_root, split=nvidia_test)
    test_data = build_dali_loader(test_pipe, len(nvidia_test))

   # Models
    print("Loading models")
    classifier = get_classifier(dataset=args.dataset)
    classifier = classifier(2).to(device)
    weights = torch.load(
        f'{args.load_path}',
        map_location=device
    )
    classifier.load_state_dict(weights)
    classifier.eval()
    classifier.requires_grad_(False)
    
    # TODO fix hard parameter
    input_size=(3,224,224)
    if args.autoencoder == 'old':
        print('Using old autoencoder')
        autoencoder= ConvWithConn().to(device)
    elif args.autoencoder == 'unet':
        print('Using unet autoencoder')
        autoencoder = AutoEncoder(input_size, encoder=classifier, device=device).to(device)
    
    autoencoder.eval()
    autoencoder.requires_grad_(False)

    torch.backends.cudnn.benchmark = True

   # Optimizers and Criterions
    print('Defining criterions')
    print(f'Using dynamic loss: {args.dyn}')
    optimizer = optim.Adam(autoencoder.parameters(), lr = args.lr)
    criterion_class = nn.CrossEntropyLoss()
    criterion_features = ReconstructionContrastiveLoss()
    criterion_recon = ReconstructionLoss(nn.MSELoss(), args.recon)
    criterions = [criterion_class, criterion_features, criterion_recon]

    if args.dataset == 'celeba':
        transform_batch = transforms.Compose([
            lambda x: x.permute(0,3,1,2),
            lambda x: x/255,
            lambda x: x.type(torch.float32),
            transforms.Resize((200, 200)),
        ])   
    #keep 200x200 for utkface
    if args.dataset == 'utkface':
        transform_batch = transforms.Compose([
            lambda x: x.permute(0,3,1,2),
            lambda x: x/255,
            lambda x: x.type(torch.float32),
        ])

    #evaluate
    epoch_metrics = defaultdict(list)
    epoch_correct = []
    epoch_bias = []
    epoch_loss = []
    epoch_loss_class = []
    epoch_loss_features = []
    epoch_loss_rec = []

    data_loader = test_data

    #tqdm for progress bar
    pbar = tqdm(data_loader, ncols=120)
    pbar.set_description_str(desc='Test')

    for idx, data in enumerate(pbar):
        image, bias_labels = data[0]["image"], data[0]["bias_and_labels"]
        bias_labels = bias_labels.squeeze()

        labels = bias_labels.clone()
        bias = bias_labels.clone()

        for i in range(bias_labels.size(0)):
            labels[i], bias[i] = nvidia_test.unmap(bias_labels[i].item())

        image = transform_batch(image)
        labels = labels.type(torch.LongTensor).to(device)
        bias = bias.to(device)

        unbiased, losses, correct, bias = iter(
            autoencoder,
            classifier,
            [image, labels, bias],
            criterions,
            args,
        ) 

        if idx == 0:   #saving autoencoder input and output images from first batch
            board.add_grid(prior='Test', ae_input=image)
            board.add_grid(prior='Test', ae_output=unbiased)

        epoch_loss.append(losses[0].item())
        epoch_loss_class.append(losses[1].item())
        epoch_loss_features.append(losses[2].item())
        epoch_loss_rec.append(losses[3].item())
        epoch_correct.extend(correct.tolist())
        epoch_bias.extend(bias.tolist())

        pbar.set_postfix_str(
            s=f'Mean loss: {np.mean(epoch_loss).mean():.4f}',
            refresh=True
        )

    #getting metrics from epoch_correct and epoch_bias
    if args.dataset == 'celeba':
        epoch_metrics = get_metrics_genderbias_cl(epoch_correct, epoch_bias)
    if args.dataset == 'utkface':
        epoch_metrics = get_metrics_racebias_cl(epoch_correct, epoch_bias)
    epoch_metrics['Loss'] = np.mean(epoch_loss)
    epoch_metrics['Loss_class'] = np.mean(epoch_loss_class)
    epoch_metrics['Loss_features'] = np.mean(epoch_loss_features)
    epoch_metrics['Loss_rec'] = np.mean(epoch_loss_rec)
    board.add_scalars(prior='Test', **epoch_metrics) 
    
    #print metrics
    print(epoch_metrics)
    board.close()

