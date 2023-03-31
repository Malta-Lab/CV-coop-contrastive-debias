from functools import partial
from typing import Any, Callable, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
from torchvision.datasets.celeba import CelebA as celeba_original
import torchvision.transforms as transforms
from tqdm import tqdm
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from typing import Any, Callable, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import os

class CustomCelebA(celeba_original):
    def __init__(
        self,
        root: str,
        split: str = "train",
        target_type: Union[List[str], str] = "attr",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,    
    ):    
        super().__init__(root, split, target_type, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        #X = 0  #PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(f'Target type "{t}" is not recognized.')

        # if self.transform is not None:
        #     X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return target

class CelebA(Dataset):

    dali_map = {
        '00': 0,
        '01': 1,
        '10': 2,
        '11': 3
    }    

    def __init__(
        self,
        root: str,
        split: str,
        target: str,
        bias: str, 
        biased: str,
        bias_prop: str,
        seed: int,
        attributes_path: str,
        transform: Optional[Callable] = None,
        download: bool = True
    ):
        
        #super().__init__(root='./datasets/')
        
        if split not in ['train', 'valid', 'test']:
            raise Exception("Split should be in ['train', 'valid', 'test'].")     

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.mul(x, 255)),
            ])

        self.dataset = CustomCelebA(
            root=root,
            split=split,
            target_type='attr',
            download=download
        )

        df_attributes = pd.read_csv(attributes_path, sep=';')
        
        #remove from df_attributes instances not included in self.dataset
        df_attributes = df_attributes[df_attributes['file_name'].isin(self.dataset.filename)]

        #if biased is a bollean str convert to bool
        
        
        #for unbiased, just use the original dataset
        if biased == 'False':
            df_merged = df_attributes   

        #if biased create a biased dataset with the desired target and bias attributes
        else:
            #TODO inserir possibilidade de viés negativo (1 para target e -1 para bias)
            target_and_bias = df_attributes[(df_attributes[target] == 1) & (df_attributes[bias] == 1)]
            nottarget_and_notbias = df_attributes[(df_attributes[target] == -1) & (df_attributes[bias] == -1)]

            target_and_notbias = df_attributes[(df_attributes[target] == 1) & (df_attributes[bias] == -1)]
            nottarget_and_bias = df_attributes[(df_attributes[target] == -1) & (df_attributes[bias] == 1)]

            if biased == 'True':
                
                largesize = min(len(target_and_bias), len(nottarget_and_notbias))
                smallsize = int(np.ceil(largesize * (bias_prop/(1-bias_prop))))
                
                target_and_bias = target_and_bias.sample(largesize, random_state=seed)
                nottarget_and_notbias = nottarget_and_notbias.sample(largesize, random_state=seed)

                target_and_notbias = target_and_notbias.sample(smallsize)
                nottarget_and_bias = nottarget_and_bias.sample(smallsize)
                
            if biased == 'equal_splits':
                
                splitsize = min(len(target_and_bias), len(nottarget_and_notbias), len(target_and_notbias), len(nottarget_and_bias))
                
                target_and_bias = target_and_bias.sample(splitsize, random_state=seed)
                nottarget_and_bias = nottarget_and_bias.sample(splitsize, random_state=seed)
                target_and_notbias = target_and_notbias.sample(splitsize, random_state=seed)
                nottarget_and_notbias = nottarget_and_notbias.sample(splitsize, random_state=seed)

            frames = [target_and_bias, nottarget_and_notbias, target_and_notbias, nottarget_and_bias]
            
            df_merged = pd.concat(frames)
           
        print('Biased:', biased)
        if biased not in ['False', False]:
            print('Bias prop:', bias_prop)
        print('N of instances in', split, ":", len(df_merged))

        #making target, bias and file_name lists
        list_target = df_merged[target] == 1
        list_target = list_target.astype(int)

        list_bias = df_merged[bias] == 1
        list_bias = list_bias.astype(int)
        
        df_filename = df_merged["file_name"]
        #turn df_filename into list
        list_filename = df_filename.values.tolist()
        
        self.label_bias = [self.map(label_, bias_) for label_, bias_ in zip(list_target, list_bias)]
        self.list_target = list_target
        self.list_bias = list_bias
        self.list_filename = list_filename

    def __len__(self) -> int:
        return len(self.list_filename)

    def map(self, bias: int, label: int) -> int:
        #return self.dali_map[str(bias.item()) + str(label.item())]   
        return self.dali_map[str(bias) + str(label)]   

    def unmap(self, value:int) -> Tuple[int, int]:
        '''return bias and label'''
        dali_map = {value: key for key, value in self.dali_map.items()}
        x = dali_map[value]
        return int(x[0]), int(x[1])


def get_dali_loaders(batch_size) -> Tuple[
    Callable[[str, CelebA], Tuple[torch.Tensor, torch.Tensor]],
    Callable[[str, CelebA], Tuple[torch.Tensor, torch.Tensor]]
]:
    from nvidia.dali import pipeline_def
    from nvidia.dali import fn
    
    @pipeline_def
    def dali_fn(file_root, split) -> Tuple[torch.Tensor, torch.Tensor]:
        jpegs, labels = fn.readers.file(
            file_root=file_root, 
            random_shuffle=True, 
            files=split.list_filename, 
            labels=split.label_bias
        )
        images = fn.decoders.image(jpegs, device='mixed')
        return images, labels

    train_fn = partial(dali_fn, batch_size=batch_size, num_threads=1, device_id=0, seed=1234)
    valid_fn = partial(dali_fn, batch_size=batch_size // 4, num_threads=1, device_id=0, seed=1234)
    return train_fn, valid_fn


def build_dali_loader(pipeline: Any, size: int) -> DALIGenericIterator:
    pipeline.build()
    return DALIGenericIterator(pipeline, ['image', 'bias_and_labels'], size=size)

# def get_dataset(args, split:str, is_biased:str) -> Dataset:
#     if args.dataset == 'celeba':
#             return CelebA(
#             root = args.dataset_path,
#             split = split,
#             target = args.target,
#             bias = args.bias,
#             biased = is_biased,
#             seed = args.seed,
#             attributes_path = os.path.join(args.dataset_path, 'celeba', 'list_attr_celeba.csv'),
#             download = True
#         )

# if __name__ == '__main__':
    
#     import parse
    
#     args.dataset = 'celeba'
#     # define pipelines
#     if args.dataset == 'celeba':
#         file_root= os.path.join(args.dataset_path, 'celeba', 'img_align_celeba')
#     if args.dataset == 'utkface':
#         file_root= os.path.join(args.dataset_path, 'UTKface', 'data')
#     train_pipe, valid_pipe = get_dali_loaders(args.batch_size)
#     train_pipe = train_pipe(file_root=file_root, split=nvidia_train)
#     valid_pipe = valid_pipe(file_root=file_root, split=nvidia_valid)
#     train_data = build_dali_loader(train_pipe, len(nvidia_train))
#     valid_data = build_dali_loader(valid_pipe, len(nvidia_valid))
    