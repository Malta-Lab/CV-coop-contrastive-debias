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
from torchvision import transforms
from tqdm import tqdm

from loaders import CelebA, build_dali_loader, get_dali_loaders
from models.autoencoder import get_autoencoder
from models.classifier import get_classifier
from losses import ReconstructionContrastiveLoss
from losses import ReconstructionLoss
from utils.general import set_seeds, get_dataset
from utils import define_weight, rec_ae_train
from utils.metrics import get_metrics_binarybias_cl, get_metrics_racebias_cl

def get_args():
    parser = argparse.ArgumentParser(description='Parameters for AE train')

    parser.add_argument('--classifier',     default='resnet', choices=['simple', 'unet', 'resnet'], type=str)
    parser.add_argument('--autoencoder',    default='old',  choices=['old', 'unet', 'resunet'], type=str)
    parser.add_argument('--feat_layer',     default='preconv', choices=['img', 'preconv', 'layer1', 'layer2', 'layer3', 'layer4', 'all'], type=str)
    parser.add_argument('--batch_size',     default=128, type=int, help='Size for each mini batch.')
    parser.add_argument('--dataset',        default='celeba', choices=['mnist', 'celeba', 'utkface'], type=str)
    parser.add_argument('--dataset_path',   default='./dataset/', type=str)
    parser.add_argument('--transform',      default='default', choices=['default', '220-180'], type=str, help='Transform size for the images')
    parser.add_argument('--lr',             default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--epoch',          default=100, type=int)
    parser.add_argument('--early_stop',     default=15, type=int)
    parser.add_argument('--gpu',            default='-1', help='GPU number')
    parser.add_argument('--load_path_cl',   default='./checkpoints/celeba/', type=str, help='Path to load the classifier.')
    parser.add_argument('--load_path_ae',   default='None', type=str, help='Path to load the autoencoder.')
    parser.add_argument('--name',           default='test', help='Run name on Tensorboard.')
    parser.add_argument('--temperature',    default=0.5, type=float, help='Temperature for the Contrastive loss')
    parser.add_argument('--alpha',          default=1, type=float, help='Weight for the class loss')
    parser.add_argument('--theta',          default=1, type=float, help='Weight for the feature loss')
    parser.add_argument('--recon',          default=0.5, type=float, help='Weight for the reconstruction loss')
    parser.add_argument('--rec_epoch',      default=3, type=int, help='Number of epochs to use the reconstruction loss')
    parser.add_argument('--dyn',            default=False, action='store_true', help='Use dynamic contrastive loss')
    parser.add_argument('--target',         default='gender', type=str, help='Target attribute')
    parser.add_argument('--bias',           default='race', type=str, help='Bias attribute')
    parser.add_argument('--seed',           default=1, type=int, help='Seed for the experiment')
    parser.add_argument('--is_biased',      default='False', type=str, help='If the train dataset is biased or not')
    parser.add_argument('--bias_prop',      default=0.1, type=float, help='Proportion of biased data')

    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    return args

def iter(
    ae: nn.Module, 
    classifier: nn.Module,
    feat_layer: str,
    mini_batch: List[torch.Tensor], 
    optimizer: nn.Module, 
    criterions: List[nn.Module],
    args: Namespace,
    train: bool
) -> Dict[str, float]:
    
    optimizer.zero_grad()
    if train:
        ae.train()
        ae.requires_grad_(True)
             
    else:
        ae.eval()
        ae.requires_grad_(False)
    
    images, labels, bias = mini_batch
    unbiased = ae(images)

    #forward and getting features from the unbiased input classifier
    features, pred = classifier(x=unbiased, output_features=True, layer=feat_layer)

    #getting features from the biased input classifier
    with torch.no_grad():
        gt_features, _ = classifier(x=images, output_features=True, layer=feat_layer)
    
    #getting criterions and calculating losses
    criterion_class, criterion_features, criterion_rec = criterions 
    loss_class = criterion_class(pred, labels)
    loss_features = criterion_features(features, gt_features, pred, labels)
    loss_rec = criterion_rec(unbiased, images)

    loss = (args.alpha * loss_class) + (args.theta * loss_features) + (args.recon * loss_rec)
    losses = [loss, loss_class, loss_features, loss_rec] #loss, class, features, reconstruction
    
    #backpropagation
    if train:
        loss.backward()
        optimizer.step()

    #correct predictions
    prediction = torch.argmax(pred, dim=1)

    return unbiased, losses, prediction, bias

if __name__ == "__main__":
    
    args = get_args()
    print(f'Starting script. Name: {args.name}')
    set_seeds(args.seed)
    
    #replace dot with underscore in the name
    bias_prop = str(args.bias_prop).replace('.','_')
    print(f'bias_prop: {bias_prop}')
    exit()
    
    #get rec_only autoencoder or train one
    if args.load_path_ae not in [None, 'None']:
        print('pre-trained autoencoder defined on args, loading pre-trained autoencoder for reconstruction')
        rec_ae_path = args.load_path_ae
            
    else:
        
        check_file=f'/checkpoints/rec_ae/{args.target}_{args.bias}_bp{bias_prop}/rec_only_model.ckpt'
        if os.path.isfile(check_file):
            print(f'pre-trained autoencoder found on {check_file}, loading pre-trained autoencoder for reconstruction')
            rec_ae_path=f'/checkpoints/rec_ae/{args.target}_{args.bias}_bp{bias_prop}/rec_only_model.ckpt'
            
        #if no pre-trained autoencoder for reconstruction is found, train one
        print ('pre-trained autoencoder not found, training autoencoder for reconstruction only')
        rec_ae_path = rec_ae_train(args=args)

    board = Tensorboard(args.name, f'./runs_ae/{args.name}', delete=True)  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # load dataset
    print('Loading Dataset:', args.dataset)
    print('Dataset path:', args.dataset_path)
    nvidia_train = get_dataset(args, split='train', is_biased=args.is_biased, bias_prop = args.bias_prop)
    nvidia_valid = get_dataset(args, split='valid', is_biased='equal_splits', bias_prop = args.bias_prop)

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
        
    # Autoencoder
    autoencoder = get_autoencoder(args.autoencoder)
    autoencoder = autoencoder().to(device)
    print(f'Loading ae weights from: {rec_ae_path}')
    weights_ae = torch.load(
        rec_ae_path,
        map_location=device
    )
    autoencoder.load_state_dict(weights_ae)
    autoencoder.train()
    autoencoder.requires_grad_(True)

    # Optimizers and Criterions
    print('Defining optimizer and criterions')
    print(f'Using dynamic loss: {args.dyn}')
    print(f'Using reconstruction loss: {args.recon} till epoch {args.rec_epoch}')
    print(f'Using Learning Rate: {args.lr}')
    print(f'Contrastive loss using features from {args.feat_layer}')

    optimizer = optim.Adam(autoencoder.parameters(), lr = args.lr)
    criterion_class = nn.CrossEntropyLoss()
    criterion_features = ReconstructionContrastiveLoss()
    criterion_recon = ReconstructionLoss(nn.MSELoss())
    criterions = [criterion_class, criterion_features, criterion_recon]

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
            board.close(), exit()
        # setting loss_rec to 0 when it reaches the specified epoch
        if epoch == args.rec_epoch:
            args.recon = 0
            print(f'Stopping reconstruction loss at epoch {epoch}')

        for is_train, description in zip([True, False], ["Train", "Valid"]):
            epoch_metrics = defaultdict(list)
            epoch_pred = []
            epoch_bias = []
            epoch_loss = []
            epoch_loss_class = []
            epoch_loss_features = []
            epoch_loss_rec = []
            epoch_labels = []
            
            if is_train and epoch == 0:
                continue

            data_loader = train_data if is_train else valid_data
            
            pbar = tqdm(data_loader, ncols=120)
            pbar.set_description_str(desc=f'{description} {epoch}', refresh=True)

            for idx, data in enumerate(pbar):
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

                unbiased, losses, prediction, bias = iter(
                    autoencoder,
                    classifier,
                    args.feat_layer,
                    [image, labels, bias],
                    optimizer,
                    criterions,
                    args,
                    train=is_train
                ) 

                if idx == 0:   #saving autoencoder input and output images from first batch
                    board.add_grid(prior=description, ae_input=image)
                    board.add_grid(prior=description, ae_output=unbiased)

                #TODO should we update the weights here or at the end of the epoch?
                if args.dyn == True and is_train == True:
                    batch_acc = sum(prediction == labels) / len(labels)
                    weight = define_weight(acc=batch_acc, n_classes=2)
                    args.alpha = max(0.1, weight)
                    args.theta = min(0.9, 1 - weight)
                
                epoch_loss.append(losses[0].item()) #loss, class, features, reconstruction
                epoch_loss_class.append(losses[1].item())
                epoch_loss_features.append(losses[2].item())
                epoch_loss_rec.append(losses[3].item())
                epoch_pred.extend(prediction.tolist())
                epoch_bias.extend(bias.tolist())
                epoch_labels.extend(labels.tolist())
                         
                pbar.set_postfix_str(
                    s=f'Mean loss: {np.mean(epoch_loss).mean():.4f}',
                    refresh=True
                )
           
            #getting metrics from epoch_correct and epoch_bias
            if args.dataset == 'celeba':
                epoch_metrics = get_metrics_binarybias_cl(epoch_labels, epoch_pred, epoch_bias)
            if args.dataset == 'utkface':
                epoch_metrics = get_metrics_racebias_cl(epoch_pred, epoch_bias)
            #adding metrics to tensorboard
            epoch_metrics['Loss'] = np.mean(epoch_loss)
            epoch_metrics['Loss_class'] = np.mean(epoch_loss_class)
            epoch_metrics['Loss_features'] = np.mean(epoch_loss_features)
            epoch_metrics['Loss_rec'] = np.mean(epoch_loss_rec)
            epoch_metrics['final alpha'] = args.alpha
            epoch_metrics['final theta'] = args.theta
            epoch_metrics['recon'] = args.recon
            # for key in dictionary, get mean of all values, FIXED
            board.add_scalars(prior=description, **epoch_metrics) 

        #a partir daqui só se aplica para validação  
        if best_acc < epoch_metrics['Acc']:
            print('Best epoch. Valid acc:', epoch_metrics['Acc'], 'Saving model...') 
            print(epoch_metrics) 
            early_stop = 0
            best_acc = epoch_metrics['Acc']
            model_path = os.path.join('./checkpoints', 'autoencoder', args.name)
            if not os.path.exists(model_path):
                os.makedirs(model_path)

            torch.save(autoencoder.state_dict(), f'{model_path}/ae_best_model.ckpt')

        else:
            early_stop += 1

        board.step()
    board.close()
    #-------------------------------------------------------------------------------