import os, math, random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity 

def initialize_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)



def fix_seed(SEED=42):
    os.environ['SEED'] = str(SEED)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    np.random.seed(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(SEED)

def psnr(label, outputs, max_val=1.):
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    img_diff = outputs - label
    rmse = math.sqrt(np.mean((img_diff)**2))
    if rmse == 0: # label과 output이 완전히 일치하는 경우
        return 100
    else:
        psnr = 20 * math.log10(max_val/rmse)
        return psnr


def ssim(label, outputs):
    label_tmp = label.squeeze(1).cpu().detach().numpy()
    outputs_tmp = outputs.squeeze(1).cpu().detach().numpy()
    ssim_value = structural_similarity(label_tmp, outputs_tmp, min=0, max=1) # win_size=5,

    return ssim_value


def loss_plot(train_losses, valid_losses, dst_path):
    plt.clf()
    # Loss Plot
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(dst_path + '/loss_plot.png')  # Loss 그래프를 이미지 파일로 저장
    plt.show()

def psnr_plot(train_psnr, valid_psnr, dst_path):
    plt.clf()
    # AUC Plot
    plt.plot(train_psnr, label='Training PSNR')
    plt.plot(valid_psnr, label='Validation PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('Training and Validation PSNR')
    plt.legend()
    plt.savefig(dst_path + '/psnr_plot.png')  # Loss 그래프를 이미지 파일로 저장
    plt.show()


def ssim_plot(train_ssim, valid_ssim, dst_path):
    plt.clf()
    # AUC Plot
    plt.plot(train_ssim, label='Training SSIM')
    plt.plot(valid_ssim, label='Validation SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('Training and Validation SSIM')
    plt.legend()
    plt.savefig(dst_path + '/ssim_plot.png')  # Loss 그래프를 이미지 파일로 저장
    plt.show()




class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, gen_path='/home/hjkim/projects/local_dev/dental_SRNet/SRGAN/checkpoint/generator.pth', disc_path='/home/hjkim/projects/local_dev/dental_SRNet/SRGAN/checkpoint/discriminator.pth'):
        self.counter = 0            
        self.patience = patience   
        self.verbose = verbose      
        self.best_score = None      
        self.early_stop = False     
        self.delta = delta          
        self.gen_path = gen_path    
        self.disc_path = disc_path  
    
    def __call__(self, gen_loss, generator, discriminator):
        if self.best_score is None:
            self.best_score = gen_loss
            self.save_checkpoint(generator, discriminator)
        elif gen_loss >= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = gen_loss
            self.save_checkpoint(generator, discriminator)
            self.counter = 0
    
    def save_checkpoint(self, generator, discriminator):
        if self.verbose:
            print(f'Generator Loss Decreased ({self.best_score:.6f}).  Saving model ...')
        torch.save(generator.state_dict(), self.gen_path)
        torch.save(discriminator.state_dict(), self.disc_path)
