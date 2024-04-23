import torch
import torch.nn as nn
from torchvision.models import vgg19, resnet50
import timm

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_activation=True, use_BatchNorm=True, **kwargs):
        super(ConvBlock, self).__init__()
        self.use_activation = use_activation
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs)

        self.bn = nn.BatchNorm2d(out_channels) if use_BatchNorm else nn.Identity()
        self.ac = (nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x1 = self.cnn(x)
        x2 = self.bn(x1)
        x3 = self.ac(x2)

        return x3 if self.use_activation else x2

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(UpsampleBlock, self).__init__()
        # self.conv = nn.Conv2d(in_channels, in_channels*scale_factor**2, kernel_size=2, stride=1, padding=1)
        self.conv = nn.Conv2d(in_channels, in_channels * scale_factor ** 2, kernel_size=2, stride=1, padding=1)  
        self.ps = nn.PixelShuffle(scale_factor)
        self.ac = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.ps(out)
        out = self.ac(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.b1 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3, 
            stride=1,
            padding=1
        )

        self.b2 = ConvBlock(
            in_channels, 
            in_channels,
            kernel_size=3, 
            stride=1,
            padding=1,
            use_activation=False
        )
    
    def forward(self, x):
        out = self.b1(x)
        out = self.b2(out)
        return out + x

class Generator(nn.Module):
    def __init__(self, in_channels=3, num_filters=64, num_blocks=5):
        super(Generator, self).__init__()
        self.initial =  nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        self.res = nn.Sequential(*[ResidualBlock(num_filters) for i in range(num_blocks)])
        self.conv = ConvBlock(num_filters, num_filters, kernel_size=3, stride=1, padding=1, use_activation=False)
        # Upsampling (2 layers)
        upsample_layers = []
        for _ in range(2):
            upsample_layers += [UpsampleBlock(num_filters, scale_factor=2)]
        self.up = nn.Sequential(*upsample_layers)
    
        # self.final =  nn.Sequential(
        #     nn.Conv2d(num_filters, in_channels, kernel_size=9, stride=1, padding=1),
        #     nn.Tanh()
        # )

        self.final = nn.Conv2d(num_filters, in_channels, kernel_size=9, stride=1, padding=1) # 1 Upsampling: 5, 1, 1 / 2 Upsampling: 9, 1, 1
    
    def forward(self, x):
        x = self.initial(x)
        out = self.res(x)
        out = self.conv(out) + x    # skip connection
        out = self.up(out)
        out = self.final(out)
        # out = torch.sigmoid(out)

        return out

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super(Discriminator, self).__init__()
        # Discriminator Blocks
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(in_channels, feature, kernel_size=3, stride=idx%2 + 1, padding=1, use_activation=True, use_BatchNorm=idx!=0),
            )
            in_channels = feature
        self.blocks = nn.Sequential(*blocks)

        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(512*8*8, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        out = self.blocks(x)
        out = self.mlp(out)
        out = self.sigmoid(out)
        
        return out
    
class vggL(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg = vgg19(pretrained=False).features[:18].eval().to(device)
        self.loss = nn.MSELoss()

    def forward(self, first, second):
        vgg_first = self.vgg(first)
        vgg_second = self.vgg(second)
        perceptual_loss = self.loss(vgg_first, vgg_second)
        return perceptual_loss
    
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        vgg19_model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18]) # exclude final fc layer

    def forward(self, x):
        return self.feature_extractor(x)
    

# ViT
# class FeatureExtractor(nn.Module):
#     def __init__(self, model_name='vit_base_patch16_384', pretrained=True):
#         super(FeatureExtractor, self).__init__()
    
#         self.model = timm.create_model(model_name, pretrained=pretrained)
#         self.model.patch_embed.proj = nn.Conv2d(1, self.model.patch_embed.proj.out_channels, 
#                                                 kernel_size=self.model.patch_embed.proj.kernel_size,
#                                                 stride=self.model.patch_embed.proj.stride,
#                                                 padding=self.model.patch_embed.proj.padding)
#         self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])  # Remove Last fc layer

#     def forward(self, x):
#         return self.feature_extractor(x)


# DenseNet
# class FeatureExtractor(nn.Module):
#     def __init__(self, model_name='densenet121', pretrained=True):
#         super(FeatureExtractor, self).__init__()
#         self.model_name = model_name
        
#         self.model = timm.create_model(model_name, pretrained=pretrained)
#         self.model.features.conv0 = nn.Conv2d(1, self.model.features.conv0.out_channels,
#                                                kernel_size=self.model.features.conv0.kernel_size,
#                                                stride=self.model.features.conv0.stride,
#                                                padding=self.model.features.conv0.padding)
#         self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])  # Remove Last fc layer

#     def forward(self, x):
#         return self.feature_extractor(x)
