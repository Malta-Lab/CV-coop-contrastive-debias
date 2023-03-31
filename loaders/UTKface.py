from functools import partial
from torch.utils.data import Dataset
import pandas as pd
from typing import Any, Callable, List, Optional, Tuple
import torchvision.transforms as transforms
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import torch
from sklearn.model_selection import train_test_split

class UTKface(Dataset):
    '''
    main_dir: path to the dataset
    transform: transformation to apply to the images
    delta_slice: number of slices for each side of the center slice
    subset -> list of patients that will be used in this dataset
    '''

    #age
    #TODO : add age groups
    '''
    Dali encoding:
        White men : 0
        White women: 1
        Black men: 2
        Black women: 3
        Asian men: 4
        Asian women: 5
        Indian men: 6
        Indian women: 7
        Other men: 8
        Other women: 9
    '''
    dali_map = {
        '00': 0,
        '10': 1,
        '01': 2,
        '11': 3,
        '02': 4,
        '12': 5,
        '03': 6,
        '13': 7,
        '04': 8,
        '14': 9
    }

    def __init__(
        self,
        root='./datasets/',
        split: str = "train",
        bias=None,
        label=None, 
        transform=None,
    ):
        self.root = root

        #checking if bias, labels and splits are valid
        if label not in ['gender', 'race', 'age']:
            raise ValueError('label "{label}" is not recognized.')
        if bias not in ['gender', 'race', 'age']:
            raise ValueError('bias "{bias}" is not recognized.')

        #reading the csv file with dataset information
        df=pd.read_csv(f'{root}UTKface/UTKface.csv')

        #randomizing the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        #putting labels and bias into df according to input
        df['label'] = df[label].apply(lambda x: int(x))
        df['bias'] = df[bias].apply(lambda x: int(x))

        #TODO map function that work with other stuff that not only classify gender with racial bias
        #mapping the labels and bias into dali format
        df['label_bias'] = [self.map(x, y) for x, y in zip(df['label'], df['bias'])]

        #split dataset into training, valid and test
        train_size = int(0.7 * len(df))
        valid_size = int(0.15 * len(df))
        test_size = len(df) - train_size - valid_size

        train_df, valtest_df = train_test_split(df, test_size=valid_size+test_size, random_state=42, stratify=df['label_bias'])
        val_df, test_df = train_test_split(valtest_df, test_size=test_size, random_state=42)

        if split == "train":
            self.set = train_df
        elif split == "valid":
            self.set = val_df
        elif split == "test":
            self.set = test_df
        else:
            raise ValueError(f'Split "{split}" is not recognized.')
               
        #inverting gender label so that women=0 and men=1 (for consistency with other datasets)
        if label == 'gender':
            self.set[label]=self.set[label].apply(lambda x: 1-x if x in [0,1] else x)

        #remove all instances with age < 15
        self.set = self.set[self.set['age'] >= 15]

        #get image names, bias and labels
        self.list_filename = self.set['imagename'].tolist()
        self.list_bias = self.set['bias'].tolist()
        self.list_label = self.set['label'].tolist()
        self.label_bias = self.set['label_bias'].tolist()

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.mul(x, 255)),
            ])

    def __len__(self) -> int:
        return len(self.list_filename)

    def __len__(self) -> int:
        return len(self.list_filename)

    def map(self, bias: int, label: int) -> int:
        return self.dali_map[str(bias) + str(label)]   

    def unmap(self, value:int) -> Tuple[int, int]:
        '''return bias and label'''
        dali_map = {value: key for key, value in self.dali_map.items()}
        x = dali_map[value]
        return int(x[0]), int(x[1])

def get_dali_loaders(batch_size) -> Tuple[
    Callable[[str, UTKface], Tuple[torch.Tensor, torch.Tensor]],
    Callable[[str, UTKface], Tuple[torch.Tensor, torch.Tensor]]
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

def get_dataset(args, split: str) -> Dataset:
    return UTKface(
    root = './datasets/',
    split = split,
    bias = 'race',
    label = 'gender'
    )

        
if __name__ == '__main__':

    nvidia_train = get_dataset(0,split='train')
    nvidia_valid = get_dataset(0,split='valid')

    train_pipe, valid_pipe = get_dali_loaders(32)
    train_pipe = train_pipe(file_root='/D/mattjie/adversarial-bias/datasets/UTKface/data', split=nvidia_train)
    valid_pipe = valid_pipe(file_root='/D/mattjie/adversarial-bias/datasets/UTKface/data', split=nvidia_valid)

    train_loader = build_dali_loader(train_pipe, len(nvidia_train))
    valid_loader = build_dali_loader(valid_pipe, len(nvidia_valid))

    print(len(train_loader))
    print(len(valid_loader))

