import os, warnings
import numpy as np
import imageio

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim as optim
from torchvision.utils import save_image
from torchvision import transforms

from model_gan_re import *
from datasets import SRdataset
from util import *
from losses import PerceptualLoss_VGG, PerceptualLoss_ResNet, ssim_loss
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings('ignore')

def train_one_epoch(loader, disc, gen, feature_extractor, opt_gen, opt_disc, criterion, criterion_content, device):
    gen.train(); disc.train()
    gen_loss, disc_loss, running_psnr = 0.0, 0.0, 0.0

    for batch_idx, (image, label) in enumerate(loader):

        low_res = image.to(device)
        high_res = label.to(device)

        fake = gen(low_res)
        # Train Discriminator
        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())
        disc_loss_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        loss_D = disc_loss_real + disc_loss_fake

        opt_disc.zero_grad()
        loss_D.backward()
        opt_disc.step()
        
        # Train Generator
        disc_fake = disc(fake)
        adversarial_loss = 1e-3*criterion(disc_fake, torch.ones_like(disc_fake))
        # Content Loss
        gen_features = feature_extractor(fake)
        real_features = feature_extractor(high_res)

        loss_content = criterion_content(gen_features, real_features)
        loss_G = loss_content + adversarial_loss

        opt_gen.zero_grad()
        loss_G.backward()
        opt_gen.step()

        gen_loss += loss_G.item()
        disc_loss += loss_D.item()
    
        batch_psnr = psnr(high_res, fake)
        
        running_psnr += batch_psnr

    final_gen_loss = gen_loss / (batch_idx+1)
    final_disc_loss = disc_loss / (batch_idx+1)
    final_psnr = running_psnr / (batch_idx+1)

    return final_gen_loss, final_disc_loss, final_psnr


def valid_one_epoch(loader, disc, gen, feature_extractor, criterion, criterion_content, device):
    gen.eval(); disc.eval()
    gen_loss, disc_loss, running_psnr = 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(loader):

            low_res = image.to(device)
            high_res = label.to(device)

            fake = gen(low_res)
            # Train Discriminator
            disc_real = disc(high_res)
            disc_fake = disc(fake.detach())

            disc_loss_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

            loss_D = disc_loss_real + disc_loss_fake
            
            # Train Generator
            disc_fake = disc(fake)
            adversarial_loss = 1e-3*criterion(disc_fake, torch.ones_like(disc_fake))
            gen_features = feature_extractor(fake)
            real_features = feature_extractor(high_res)

            loss_content = criterion_content(gen_features, real_features)
            loss_G = loss_content + adversarial_loss

            gen_loss += loss_G.item()
            disc_loss += loss_D.item()
        
            batch_psnr = psnr(high_res, fake)
            running_psnr += batch_psnr

        final_gen_loss = gen_loss / (batch_idx+1)
        final_disc_loss = disc_loss / (batch_idx+1)
        final_psnr = running_psnr / (batch_idx+1)

    return final_gen_loss, final_disc_loss, final_psnr



def valid_test(loader, disc, gen, feature_extractor, criterion, criterion_content, device, output_dir="/home/hjkim/projects/local_dev/dental_SRNet/SRGAN/result_gan_re"):
    os.makedirs(output_dir, exist_ok=True)  # 결과를 저장할 디렉토리 생성
    gen.eval(); disc.eval()
    min_value, max_value =  -1024, 3071
    gen_loss, disc_loss, running_psnr = 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(loader):

            low_res = image.to(device)
            high_res = label.to(device)

            fake = gen(low_res)
            # Train Discriminator
            disc_real = disc(high_res)
            disc_fake = disc(fake.detach())

            disc_loss_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

            loss_D = disc_loss_real + disc_loss_fake
            
            # Train Generator
            disc_fake = disc(fake)
            adversarial_loss = 1e-3*criterion(disc_fake, torch.ones_like(disc_fake))
            gen_features = feature_extractor(fake)
            real_features = feature_extractor(high_res)

            loss_content = criterion_content(gen_features, real_features)
            loss_G = loss_content + adversarial_loss

            gen_loss += loss_G.item()
            disc_loss += loss_D.item()
        
            batch_psnr = psnr(high_res, fake)
            running_psnr += batch_psnr


            # Save results
            image = low_res.detach().cpu()  
            outputs = fake.detach().cpu() 
            label = high_res.detach().cpu() 

            for i in range(outputs.size(0)):
                denorm_image = ( (image[i][0,:,:] * (max_value - min_value)) + min_value )
                denorm_output = ( (outputs[i][0,:,:] * (max_value - min_value)) + min_value )
                denorm_label = ( (label[i][0,:,:] * (max_value - min_value)) + min_value )
                
                imageio.imwrite(os.path.join(output_dir, f'input_{batch_idx * test_dl.batch_size + i}.tiff'), denorm_image)
                imageio.imwrite(os.path.join(output_dir, f'output_{batch_idx * test_dl.batch_size + i}.tiff'), denorm_output)
                imageio.imwrite(os.path.join(output_dir, f'label_{batch_idx * test_dl.batch_size + i}.tiff'), denorm_label)

        final_gen_loss = gen_loss / (batch_idx+1)
        final_disc_loss = disc_loss / (batch_idx+1)
        final_psnr = running_psnr / (batch_idx+1)

    return final_gen_loss, final_disc_loss, final_psnr


def split_dataset(file_path = "/home/hjkim/projects/local_dev/dental_SRNet/data/tiff"):
       
    # Total sample
    file_list = os.listdir(file_path)
    num_sample = len(file_list)
    total_indices = list(range(num_sample))

    val_test_ratio = 0.3  

    # train과 validation, test 데이터셋을 위한 인덱스 추출
    train_indices, temp_indices = train_test_split(total_indices, test_size=val_test_ratio, random_state=42)    # 7 : 3
    val_test_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42) # 5 : 5

    train_data = [os.path.join(file_path, file_list[idx]) for idx in train_indices]
    valid_data = [os.path.join(file_path, file_list[idx]) for idx in val_test_indices]
    test_data = [os.path.join(file_path, file_list[idx]) for idx in test_indices]

    return train_data, valid_data, test_data


if __name__ == "__main__":
    
    # Environment
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = '1'

    # Options 
    EPOCHS = 200
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-5
    CHECKPOINT_PATH = "/home/hjkim/projects/local_dev/dental_SRNet/SRGAN/checkpoint"
    PLOT_PATH = "/home/hjkim/projects/local_dev/dental_SRNet/SRGAN/checkpoint"
    IMAGE_SIZE = 512

    fix_seed(42)
    device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
    # Models
    gen = Generator(in_channels=1).to(device)
    disc = Discriminator(in_channels=1).to(device)

    # Loss Function & Optimizer
    criterion = nn.MSELoss().to(device)
    # criterion = nn.BCEWithLogitsLoss().to(device)
    feature_extractor = FeatureExtractor().to(device)
    criterion_content = nn.MSELoss().to(device)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    scheduler = ReduceLROnPlateau(opt_gen, mode='min', patience=3, factor=0.5, verbose=True)

    # Import Dataset
    train_data, valid_data, test_data = split_dataset()

    train_ds = SRdataset(train_data)
    valid_ds = SRdataset(valid_data)
    test_ds = SRdataset(test_data)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, drop_last=True)

    print(f"TRAIN: {len(train_dl)}, VALID: {len(valid_dl)}, TEST: {len(test_dl)}")
    train_gen_losses, train_disc_losses, train_psnr = [], [], []
    valid_gen_losses, valid_disc_losses, valid_psnr = [], [], []
    early_stopping = EarlyStopping(patience=10, verbose=True,
                                    gen_path=os.path.join(CHECKPOINT_PATH, 'generator_re.pth'), 
                                    disc_path=os.path.join(CHECKPOINT_PATH,'discriminator_re.pth'))
    # Training 
    for epoch in range(EPOCHS):
        
        # Train
        train_epoch_gen_loss, train_epoch_disc_loss, train_epoch_psnr = train_one_epoch(train_dl, disc, gen, feature_extractor, opt_gen, opt_disc, criterion, criterion_content, device)
        
        train_gen_losses.append(train_epoch_gen_loss)
        train_disc_losses.append(train_epoch_disc_loss)
        train_psnr.append(train_epoch_psnr)

        # Valid
        valid_epoch_gen_loss, valid_epoch_disc_loss, valid_epoch_psnr = valid_one_epoch(valid_dl, disc, gen, feature_extractor, criterion, criterion_content, device)
        scheduler.step(valid_epoch_disc_loss)
        valid_gen_losses.append(valid_epoch_gen_loss)
        valid_disc_losses.append(valid_epoch_disc_loss)
        valid_psnr.append(valid_epoch_psnr)

        print(f"Epoch: {epoch+1}, Train Gen Loss: {train_epoch_gen_loss:.6f}, Train Disc Loss: {train_epoch_disc_loss:.6f}, Train PSNR: {train_epoch_psnr:.6f}, Valid Gen Loss: {valid_epoch_gen_loss:.6f}, Valid Disc Loss: {valid_epoch_disc_loss:.6f}, Valid PSNR: {valid_epoch_psnr:.6f}")

        # Save models if Discriminator Loss >= 1 
        if valid_epoch_disc_loss >= 1.0:
            break

        early_stopping(valid_epoch_gen_loss, gen, disc)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # torch.save(gen.state_dict(), os.path.join(CHECKPOINT_PATH, 'generator_re.pth'))
        # torch.save(disc.state_dict(), os.path.join(CHECKPOINT_PATH,'discriminator_re.pth'))

    # Save Plot
    # loss_plot(train_loss, valid_loss, PLOT_PATH)
    psnr_plot(train_psnr, valid_psnr, PLOT_PATH)

    # Test 
    gen = Generator(in_channels=1).to(device)
    disc = Discriminator(in_channels=1).to(device)

    gen.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH,"generator_re.pth")))
    disc.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH,"discriminator_re.pth")))

    test_epoch_gen_loss, test_epoch_disc_loss, test_epoch_psnr= valid_test(test_dl, disc, gen, feature_extractor, criterion, criterion_content, device, )
    print("FINAL PERFORMANCE")
    print(f"Test Gen Loss: {test_epoch_gen_loss:.6f}, Test Disc Loss: {test_epoch_disc_loss:.6f}, Test PSNR: {test_epoch_psnr:.6f}")
