import torch
import torch.nn as nn
from torchvision import models

# Change this for a more complex decoder netowrk
def decode_conv(in_channels, out_channels):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True))
    return conv

class ResnetUnet(nn.Module):
    def __init__(self, class_num):
        super().__init__()

        # Import model of intrest
        self.rn_model = models.resnet152(pretrained=True)
        # get diffrent layers of model
        self.rn_layers = list(self.rn_model.children())
        # Remove AdaptiveAvgPool2d and Linear layers (last 2 layers)
        # all resnets models have the same conv layer sections 8 just
        # each layer in the sections are deeper. 

        # Input shape [bs, 3, h, w]
        # Layers 0 -> 2 all output [bs, 64, h/2, w/2]
        # Layers 3 -> 4 all output [bs, 64, h/4, w/4]
        # Layer 5 output [bs, 128, h/8, w/8]
        # layer 6 output [bs, 256, h/16, w/16]
        # layer 7 output [bs, 512, h/32, w/32]
        
        # Make encoder layers
        self.encoder_layer0 = nn.Sequential(*self.rn_layers[:3])
        self.encoder_layer1 = nn.Sequential(*self.rn_layers[3:5])
        self.encoder_layer2 = nn.Sequential(self.rn_layers[5])
        self.encoder_layer3 = nn.Sequential(self.rn_layers[6])
        self.encoder_layer4 = nn.Sequential(self.rn_layers[7])
        
        # Decoder layers and expansion layers
        # After expansion cat corisponing layer (decoder0 with encoder3)
        self.expand0 = nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=2, stride=2)
        self.decoder_layer0 = decode_conv(2048, 1024)

        self.expand1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.decoder_layer1 = decode_conv(1024, 512)

        self.expand2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.decoder_layer2 = decode_conv(512, 256)
        
        self.expand3 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=2, stride=2)
        self.decoder_layer3 = decode_conv(128,64)

        self.expand4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2)
        self.out = decode_conv(64, class_num)
        
        
    def forward(self, x):
        # Encoder
        layer0 = self.encoder_layer0(x)
        layer1 = self.encoder_layer1(layer0)
        layer2 = self.encoder_layer2(layer1)
        layer3 = self.encoder_layer3(layer2)
        layer4 = self.encoder_layer4(layer3)

        # Decoder and expansions
        x = self.expand0(layer4)
        x = torch.cat((x, layer3), dim=1)
        x = self.decoder_layer0(x)

        x = self.expand1(x)
        x = torch.cat((x, layer2), dim=1)
        x = self.decoder_layer1(x)

        x = self.expand2(x)
        x = torch.cat((x, layer1), dim=1)
        x = self.decoder_layer2(x)

        x = self.expand3(x)
        x = torch.cat((x, layer0), dim=1)
        x = self.decoder_layer3(x)

        x = self.expand4(x)
        out = self.out(x)

        return out
    
