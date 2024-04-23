import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from math import exp

class PerceptualLoss_VGG(nn.Module):
    def __init__(self):
        super(PerceptualLoss_VGG, self).__init__()
        vgg_model = models.vgg19(pretrained=True, progress=True).cuda()
        self.features = nn.Sequential(*list(vgg_model.features.children())[:35])  

    def forward(self, x, y):
        x = torch.cat([x, x, x], dim=1) # Gray scale to RGB
        y = torch.cat([y, y, y], dim=1)

        features_x = self.features(x)
        features_y = self.features(y)

        loss = torch.mean(torch.abs(features_x - features_y))
        return loss
    


class PerceptualLoss_ResNet(nn.Module):
    def __init__(self, layer_name='layer3'):
        super(PerceptualLoss_ResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.layer_name = layer_name
        self.loss = nn.MSELoss()
    
    def forward(self, x, y):
        for name, module in self.resnet._modules.items():
            x = module(x)
            y = module(y)
            if name == self.layer_name:
                break
        
        loss = self.loss(x, y)
        return loss

def create_window(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    gauss = gauss.view(1, 1, window_size, 1)  # 필터의 차원을 (1, 1, window_size, 1)로 설정

    return gauss/gauss.sum()


def ssim_loss(img1, img2, window_size=11, size_average=True, sigma=1.5):
    window = create_window(window_size, sigma)
    window = window.to(img1.device)

    # Compute mean of images
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.shape[1])

    # Compute mean of squares
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    # Compute product of means
    mu1_mu2 = mu1 * mu2

    # Compute variance of images
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.shape[1]) - mu2_sq

    # Compute covariance
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.shape[1]) - mu1_mu2

    c1 = (0.01) ** 2
    c2 = (0.03) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
