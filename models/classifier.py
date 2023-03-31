import torch
from torch import nn
import torchvision.models as models
from torchsummary import summary
from models.utils.enc_utils import Encoder

def get_classifier(classifier: str) -> nn.Module:
    if classifier == 'resnet':
        print('Using Resnet18 classifier')
        return Resnet18
    elif classifier == 'simple':
        print('Using SimpleConvNetwork classifier')
        return SimpleConvNetwork
    elif classifier == 'unet':
        print('Using UnetEncoder classifier')
        return UnetEncoder  
    else:
        raise Exception('Classifier not found')

def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    '''
    Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    '''
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x

class Resnet18(nn.Module):

    def __init__(self, classes: int, pretrained=True, fc_size: int = 512) -> None:
        super().__init__()
        self.normalize = pretrained

        self.encoder = models.resnet18(pretrained=pretrained)
        self.encoder.fc = nn.Sequential(
            nn.Linear(fc_size, fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, classes)
        )

    def forward(self, x: torch.Tensor, output_features: bool = False, layer: str = None) -> torch.Tensor:
        '''
            Forward the input through a resnet so we can get a single
            convolution layer.

            args:
                x (tensor): the input of the encoder
                output_features (bool): whether it should output a feature embedding.
        '''
        if output_features and layer is None:
            raise Exception('If output_features is True, you should inform a layer')

        if layer is not None:
            assert layer in ['img', 'preconv', 'layer1', 'layer2', 'layer3', 'layer4', 'all']

        img = normalize_imagenet(x) if self.normalize else x
        #img = x
        
        conv1 = self.encoder.conv1(img)
        x = self.encoder.bn1(conv1)
        x = self.encoder.relu(x)
        preconv = self.encoder.maxpool(x)


        layer1 = self.encoder.layer1(preconv)     # [b, 64, 56, 56]
        layer2 = self.encoder.layer2(layer1)      # [b, 128, 28, 28]
        layer3 = self.encoder.layer3(layer2)      # [b, 256, 14, 14]
        layer4 = self.encoder.layer4(layer3)      # [b, 512, 7, 7]

        #print
        feature_embedding = {
            'img': img,
            'preconv': preconv,
            'layer1': layer1,
            'layer2': layer2,
            'layer3': layer3,
            'layer4': layer4,
            'all': [img, x, layer1, layer2, layer3, layer4]
        }
        features = feature_embedding[layer] if output_features else None
    
        x = self.encoder.avgpool(layer4)
        x = torch.flatten(x, 1)
        x = self.encoder.fc(x)

        return features, x


class SimpleConvNetwork(nn.Module):
    '''
    Plain convolution network
    '''

    def __init__(self, classes: int, **kwargs):
        super().__init__()
        self.bnorm = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(64, classes)

    def forward(self, x: torch.Tensor, output_features: bool = False, layer: str = None) -> None:
        '''
            Forward the input through a resnet so we can get a single
            convolution layer.

            args:
                x (tensor): the input of the encoder
                output_features (bool): whether it should output a feature embedding.
        '''

        x = self.bnorm(x)

        conv1 = self.maxpool(self.relu(self.conv1(x)))  # [b, 32, 14, 14]
        conv2 = self.relu(self.conv2(conv1))            # [b, 32, 14, 14]
        conv3 = self.relu(self.conv3(conv2))            # [b, 64, 7, 7]
        conv4 = self.relu(self.conv4(conv3))            # [b, 64, 7, 7]

        feat_low = self.avgpool(conv4)
        feat_low = feat_low.view(feat_low.size(0), -1)  # [b, 64]
        output = self.fc(feat_low)

        feature_embedding = {
            'preconv': x,
            'conv1': conv1,
            'conv2': conv2,
            'conv3': conv3,
            'conv4': conv4,
            'all': [conv4, conv3, conv2, conv1]
        }
        features = feature_embedding[layer] if output_features else None
        return features, output

class UnetEncoder(nn.Module):

    def __init__(self, classes: int, **kwargs):
        super().__init__()

        self.encoder = Encoder() 
        self.encoder.fc = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, classes)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor, output_features: bool = False, layer: str = None) -> None:

        if output_features and layer is None:
            raise Exception('If output_features is True, you should inform a layer')

        if layer is not None:
            assert layer in ['preconv', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'all']

        features, x = self.encoder(x, output_features=output_features, layer=layer)
        #x = self.bottleneck(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.encoder.fc(x)

        return features, x

if __name__ == '__main__':
    model = Resnet18(2).to('cuda')
    #testing model
    x = torch.randn(1, 3, 224, 224).to('cuda')
    features, x = model(x)
    print(x.shape)