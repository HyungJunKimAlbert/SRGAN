import os, cv2
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from util import *
from PIL import Image


class SRdataset(Dataset):
    """
        Parameters:
            path (List): Image File Paths List 
        Returns
            Input Sample (np.array): Blurred Image
            Label Sample (np.array): Original Image
    """
    def __init__(self, paths,img_size=512):
        self.paths = paths        
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((img_size // 4, img_size // 4), Image.BICUBIC),
                transforms.ToTensor(),
                # transforms.Normalize(0.449, 0.226),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), Image.BICUBIC),
                transforms.ToTensor(),
                # transforms.Normalize(0.449, 0.226),
                # transforms.RandomHorizontalFlip(),   
                # transforms.RandomRotation(10),          
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self,idx):
        path = self.paths[idx]
        img = np.array(Image.open(path))
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32).squeeze()
        inp = cv2.GaussianBlur(img, (11, 11), 0)

        input_sample = Image.fromarray(inp)
        label_sample = Image.fromarray(img)
        # input_sample, label_sample = torch.tensor(inp, dtype=torch.float32), torch.tensor(img, dtype=torch.float32)
        
        input_sample = self.lr_transform(label_sample)  # .expand(3, -1, -1)
        label_sample = self.hr_transform(label_sample)  # .expand(3, -1, -1)

        return input_sample, label_sample
