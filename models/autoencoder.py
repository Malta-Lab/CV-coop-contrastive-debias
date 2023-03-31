import torch
from torch import nn
import torchvision.models as models
from torchsummary import summary

from models.utils.enc_utils import Decoder, Double_Conv2d, Encoder

def get_autoencoder(autoencoder: str) -> nn.Module:
    if autoencoder == 'old':
        print('Using old autoencoder')
        return ConvWithConn
    elif autoencoder == 'unet':
        print('Using Unet autoencoder')
        return Unet
    elif autoencoder == 'resunet':
        print('Using ResUnet autoencoder')
        return ResUnet18
    else:
        raise Exception('Autoencoder not found')

#old autoencoder
class ConvWithConn(nn.Module):
    """
    Convolutional Auto Encoder.
    """
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = Double_Conv2d(3, 16)
        self.conv2 = Double_Conv2d(16, 32)
        self.conv3 = Double_Conv2d(32, 64)

        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)

        self.conv5 = Double_Conv2d(64, 32)
        self.conv6 = Double_Conv2d(32, 16)

        self.last_conv = nn.Conv2d(
            in_channels=16,
            out_channels=3,
            kernel_size=1,
            padding=0
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()

        conv1 = self.conv1(x)
        pool1 = self.pool(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool(conv2)

        conv3 = self.conv3(pool2)
        up1 = self.deconv1(conv3)

        up1 = torch.cat([conv2, up1], dim=1)
        up1 = self.conv5(up1)

        up2 = self.deconv2(up1)
        up2 = torch.cat([conv1, up2], dim=1)
        up2 = self.conv6(up2)

        final_conv = self.last_conv(up2)
        return final_conv

class Unet(nn.Module):
    """
    UNet-like autoencoder
    """

    def __init__(self):
        super().__init__()

        self.encoder = Encoder() 
        self.decoder = Decoder()

        self.bottleneck = Double_Conv2d(512, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, x = self.encoder(x, output_features=True, layer='all')
        x = self.bottleneck(x)
        x = self.decoder(x, features)
        return x

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

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True)
    )

def final_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
    )


def up_conv(in_channels, out_channels):
    return nn.Sequential(nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
    ),
    nn.BatchNorm2d(out_channels),
    nn.LeakyReLU(inplace=True)
)

#unet with resnet like encoders
class ResUnet18(nn.Module):
        
    def __init__(self, pretrained=True) -> None:
        super().__init__()
        self.normalize = pretrained
        #new conv prior to resnet first conv
        self.conv0 = double_conv(3, 32)
        #new resnet first conv (with correct channels)
        self.conv1 = nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder = models.resnet18(pretrained=pretrained)
        # self.encoder.fc = nn.Sequential(
        #     nn.Linear(fc_size, fc_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(fc_size, classes)
        # )
        #self.bottleneck = double_conv(512, 512)
        self.up_conv6 = up_conv(512, 256)
        self.conv6 = double_conv(256 + 256, 256)
        self.up_conv7 = up_conv(256, 128)
        self.conv7 = double_conv(128 + 128, 128)
        self.up_conv8 = up_conv(128, 64)
        self.conv8 = double_conv(64 + 64, 64)
        self.up_conv9 = up_conv(64, 64)
        self.conv9 = double_conv(64 + 64, 64)
        #finish autoencoder
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = double_conv(32 + 32, 32)
        self.conv11 = final_conv(32, 3)
        

        #self activation layer with 256 range
        self.sigmoid = nn.Sigmoid()

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
            assert layer in ['resconv', 'conv1', 'conv2', 'conv3', 'conv4', 'all']

        img = normalize_imagenet(x) if self.normalize else x
        #img = x


        #encoding
        layer0 = self.conv0(img)
        
        resconv = self.conv1(layer0)
        x = self.encoder.bn1(resconv)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        layer1 = self.encoder.layer1(x)          # [b, 64, 56, 56]
        layer2 = self.encoder.layer2(layer1)      # [b, 128, 28, 28]
        layer3 = self.encoder.layer3(layer2)      # [b, 256, 14, 14]
        layer4 = self.encoder.layer4(layer3)      # [b, 512, 7, 7]

        #decoding
        x = self.up_conv6(layer4)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv6(x)
        x = self.up_conv7(x)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv7(x)
        x = self.up_conv8(x)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv8(x)
        x = self.up_conv9(x)
        x = torch.cat([x, resconv], dim=1)
        x = self.conv9(x)
        x = self.up_conv10(x)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv10(x)

        #return to image space - final conv
        x = self.conv11(x)

        #RGB into 256 range
        x = self.sigmoid(x)
                
        #get features
        feature_embedding = {
            'img': img,
            'layer0': layer0,
            'resconv': resconv,
            'layer1': layer1,
            'layer2': layer2,
            'layer3': layer3,
            'layer4': layer4,
            'all': [layer4, layer3, layer2, layer1, layer0, img]
        }
        features = feature_embedding[layer] if output_features else None
    
        return features, x

if __name__ == "__main__":
    # #testing ConvwithConn
    # model = ConvWithConn()
    # print(model)
    # input=torch.rand(1,3,224,224)
    # _, output = model(input)
    # print(output.shape)

    #testing AutoEncoder
    model = Unet().to('cuda')
    print(model)
    input=torch.rand(1,3,224,224).to('cuda')
    output = model(input)
    print(output.shape)

    # from torchsummary import summary
    # summary(model, (3, 224, 224))