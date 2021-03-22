import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet 


# Change this for a more complex decoder netowrk
def decode_conv(in_channels, out_channels):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True))
    return conv

class Effnetb0(nn.Module):
    def __init__(self):
        super().__init__()
        self.effnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.up1 = nn.ConvTranspose2d(1280, 640, 2, 2)
        self.decode1 = decode_conv(640, 640)
        self.up2 = nn.ConvTranspose2d(640, 320, 2, 2)
        self.decode2 = decode_conv(320, 320)
        self.up3 = nn.ConvTranspose2d(320, 160, 2, 2)
        self.decode3 = decode_conv(160, 160)
        self.up4 = nn.ConvTranspose2d(160, 80, 2, 2)
        self.decode4 = decode_conv(80, 80)
        self.up5 = nn.ConvTranspose2d(80, 20, 2, 2)
        self.out = decode_conv(20, 5)

    def forward(self, x):
        x = self.effnet.extract_features(x)
        x = self.up1(x)
        x = self.decode1(x)
        x = self.up2(x)
        x = self.decode2(x)
        x = self.up3(x)
        x = self.decode3(x)
        x = self.up4(x)
        x = self.decode4(x)
        x = self.up5(x)
        x = self.out(x)
        return x

class Effnetb7(nn.Module):
    def __init__(self):
        super().__init__()
        self.effnet = EfficientNet.from_pretrained('efficientnet-b7')
        self.up1 = nn.ConvTranspose2d(2560, 640, 2, 2)
        self.decode1 = decode_conv(640, 640)
        self.up2 = nn.ConvTranspose2d(640, 320, 2, 2)
        self.decode2 = decode_conv(320, 320)
        self.up3 = nn.ConvTranspose2d(320, 160, 2, 2)
        self.decode3 = decode_conv(160, 160)
        self.up4 = nn.ConvTranspose2d(160, 80, 2, 2)
        self.decode4 = decode_conv(80, 80)
        self.up5 = nn.ConvTranspose2d(80, 20, 2, 2)
        self.out = decode_conv(20, 5)

    def forward(self, x):
        x = self.effnet.extract_features(x)
        x = self.up1(x)
        x = self.decode1(x)
        x = self.up2(x)
        x = self.decode2(x)
        x = self.up3(x)
        x = self.decode3(x)
        x = self.up4(x)
        x = self.decode4(x)
        x = self.up5(x)
        x = self.out(x)
        return x

if __name__=="__main__":
    m = Effnetb7()
    in_x = torch.randn((5,3,512,256))
    m(in_x)

        
