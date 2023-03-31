import torch
from loaders.celeba import CelebA, build_dali_loader, get_dali_loaders
from models import get_classifier
from models.autoencoder import AutoEncoder
from models.classifier import Resnet


if __name__ == "__main__":
    # resnet: Resnet = get_classifier('celeba')
    # classifier = resnet(classes=2)

    # img_size = (3, 224, 224)
    # ae = AutoEncoder(img_size, encoder=classifier)
    # img = torch.Tensor(size=(1, *img_size))
    # ae(img)

    dataset_path = './celeba/celeba/img_align_celeba'
    train_split = CelebA(root='./celeba/', split='train', class_subset=['Arched_Eyebrows'], download=True)
    valid_split = CelebA(root='./celeba/', split='valid', class_subset=['Arched_Eyebrows'], download=True)

    train_pipe, valid_pipe = get_dali_loaders(32)
    train_pipe = train_pipe(file_root=dataset_path, split=train_split)
    valid_pipe = valid_pipe(file_root=dataset_path, split=valid_split)
    train_data = build_dali_loader(train_pipe, len(train_split))
    valid_data = build_dali_loader(valid_pipe, len(valid_split))