from typing import List
from torch import cat, nn, Tensor


class Double_Conv2d(nn.Module):
    """
    Double convolutional layers from the UNet implementation.
    """
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        self.double_conv2d = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=3,
                padding=1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=3,
                padding=1
            ),
            nn.LeakyReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.double_conv2d(x)

#Up convolution from Unet with activation
def Up_conv2d(in_channel: int, out_channel: int) -> nn.Module:
    """
    Up convolutional layer from the UNet implementation.
    """
    #include 
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=2,
            stride=2
        ),
        nn.LeakyReLU()
    )



class Encoder(nn.Module):
    """
    Encoder for the UNet
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = Double_Conv2d(3, 32)
        self.conv2 = Double_Conv2d(32, 64)
        self.conv3 = Double_Conv2d(64, 128)
        self.conv4 = Double_Conv2d(128, 256)
        self.conv5 = Double_Conv2d(256, 512)

        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor, output_features: bool = True, layer: str = None) -> Tensor:

        x = x.float()
    
        conv1 = self.conv1(x)
        pool1 = self.max_pool2d(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.max_pool2d(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.max_pool2d(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.max_pool2d(conv4)

        conv5 = self.conv5(pool4)

        feature_embedding = {
            'preconv': x,
            'conv1': conv1,
            'conv2': conv2,
            'conv3': conv3,
            'conv4': conv4,
            'conv5': conv5,
            'all': [conv5, conv4, conv3, conv2, conv1]
        }
        features = feature_embedding[layer] if output_features else None

        return features, conv5


# TODO stop concat actual image
class Decoder(nn.Module):
    """
    Decoder for the UNet
    """
    def __init__(self) -> None:
        super().__init__()
        self.up1 = Up_conv2d(512, 256)
        self.conv1 = Double_Conv2d(512, 256)
        self.up2 = Up_conv2d(256, 128)
        self.conv2 = Double_Conv2d(256, 128)
        self.up3 = Up_conv2d(128, 64)
        self.conv3 = Double_Conv2d(128, 64)
        self.up4 = Up_conv2d(64, 32)
        self.conv4 = Double_Conv2d(64, 32)
        self.conv5 = Double_Conv2d(32, 3)

    def forward(self, x: Tensor, conv: List[Tensor]) -> Tensor:

        up1 = self.up1(x)
        up1 = cat([up1, conv[1]], 1)
        conv1 = self.conv1(up1)

        up2 = self.up2(conv1)
        up2 = cat([up2, conv[2]], 1)
        conv2 = self.conv2(up2)

        up3 = self.up3(conv2)
        up3 = cat([up3, conv[3]], 1)
        conv3 = self.conv3(up3)

        up4 = self.up4(conv3)
        up4 = cat([up4, conv[4]], 1)
        conv4 = self.conv4(up4)

        conv5 = self.conv5(conv4)
        #sigmoid
        sigmoid = nn.Sigmoid()(conv5)

        return sigmoid
