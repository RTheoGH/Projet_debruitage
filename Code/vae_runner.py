"""
vae_runner.py

Single-file VAE training & evaluation runner for image denoising.

Usage (basic):
    python vae_runner.py

Configuration: edit the CONFIG section below (paths, image size, epochs, batch size, ...)
You can also override some values via command-line args (see --help).

Dependencies:
    pip install torch torchvision pillow tqdm numpy matplotlib

What this file contains:
- PairedDataset: loads input / GT images from directories (matching filenames)
- VAE model (Encoder, Decoder)
- Training loop, validation loop (computes MSE, PSNR, SSIM, DISTS)
- Checkpoint saving and sample output saving as PNG images

Assumption: you already prepared your datasets. For each split you must have two folders:
    - inputs: noisy images
    - gts: ground-truth images
Filenames must match across inputs and gts folders.
"""

import os
import argparse
from PIL import Image
import numpy as np
from math import log10, sqrt
from tqdm import tqdm
import re
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from DISTS_pt import DISTS

import random

def pick_random_indices(dataloader, num_samples=8):
    total = len(dataloader.dataset)
    return set(random.sample(range(total), k=min(num_samples, total)))

# ------------------ CONFIG------------------
TRAIN_INPUT_DIR = '../allImages/train/noised/gaussian'  
TRAIN_GT_DIR =    '../allImages/train/truth' 
VAL_INPUT_DIR =   '../allImages/validation/noised/gaussian'  
VAL_GT_DIR =      '../allImages/validation/truth'
TEST_INPUT_DIR =  '../allImages/validation/noised/gaussian/test'
TEST_GT_DIR =     '../allImages/validation/truth/test'

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
NUM_EPOCHS = 30
SAVE_EVERY = 1
NUM_WORKERS = 4

LR = 0.0002
KL_WEIGHT = 0.00025  # Max Weight for KL divergence loss

CHECKPOINT_DIR = 'model/VAE_runner'
SAMPLES_DIR = 'model/VAE_samples'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PairedImageDataset(Dataset):
    """Load pairs of images (input, gt) from two folders. Filenames must match."""
    def __init__(self, input_dir, gt_dir, img_size=(128,128), augment=False):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.augment = augment
        self.img_size = img_size
        
        def index_map(d):
            m = {}
            if not os.path.isdir(d):
                return m
            for f in os.listdir(d):
                p = os.path.join(d, f)
                if not os.path.isfile(p):
                    continue
                name = os.path.splitext(f)[0]
                match = re.search(r"(\d+)", name)
                if match:
                    idx = int(match.group(1))
                    m[idx] = f
            return m

        in_map = index_map(input_dir)
        gt_map = index_map(gt_dir)

        common = sorted(k for k in in_map.keys() if k in gt_map)
        if not common:
            # Fallback: try matching by exact filename if numeric index fails or is not preferred
            # But keeping original logic for consistency
            pass

        if not common:
             # Fallback to simple filename matching if regex fails
            in_files = set(os.listdir(input_dir))
            gt_files = set(os.listdir(gt_dir))
            common_files = sorted(list(in_files & gt_files))
            self.pairs = [(f, f) for f in common_files]
        else:
            self.pairs = [(in_map[k], gt_map[k]) for k in common]

        if len(self.pairs) == 0:
             raise SystemExit(f'No matching filenames found between {input_dir} and {gt_dir}.')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        in_fname, gt_fname = self.pairs[idx]
        p_in = os.path.join(self.input_dir, in_fname)
        p_gt = os.path.join(self.gt_dir, gt_fname)
        img_in = Image.open(p_in).convert('RGB')
        img_gt = Image.open(p_gt).convert('RGB')

        if img_in.size[0] < self.img_size[0] or img_in.size[1] < self.img_size[1]:
            img_in = TF.resize(img_in, self.img_size)
            img_gt = TF.resize(img_gt, self.img_size)

        if self.augment:
            i, j, h, w = transforms.RandomCrop.get_params(img_in, output_size=self.img_size)
            img_in = TF.crop(img_in, i, j, h, w)
            img_gt = TF.crop(img_gt, i, j, h, w)

            if random.random() > 0.5:
                img_in = TF.hflip(img_in)
                img_gt = TF.hflip(img_gt)
        else:
            img_in = TF.center_crop(img_in, self.img_size)
            img_gt = TF.center_crop(img_gt, self.img_size)

        t_in = TF.to_tensor(img_in)
        t_gt = TF.to_tensor(img_gt)

        # Normalize to [-1, 1]
        t_in = TF.normalize(t_in, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        t_gt = TF.normalize(t_gt, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        return t_in, t_gt, in_fname


# ------------------ VAE Model ------------------

class VAE(nn.Module):
    def __init__(self, img_channels=3, latent_dim=512):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1), # 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        self.flatten_size = 512 * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1), # 128x128
            nn.Tanh() # Output range [-1, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_flat = x_encoded.view(x_encoded.size(0), -1)
        
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        
        z = self.reparameterize(mu, logvar)
        
        z_decoded = self.decoder_input(z)
        z_reshaped = z_decoded.view(z_decoded.size(0), 512, 4, 4)
        
        reconstruction = self.decoder(z_reshaped)
        return reconstruction, mu, logvar

# ------------------ Utils ------------------

def compute_psnr(img1, img2, max_val=1.0):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * log10(max_val / sqrt(mse))

def compute_ssim(img1, img2, window_size=11):
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)

    mu1 = F.avg_pool2d(img1, window_size, 1, window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, 1, window_size//2)

    sigma1 = F.avg_pool2d(img1*img1, window_size, 1, window_size//2) - mu1**2
    sigma2 = F.avg_pool2d(img2*img2, window_size, 1, window_size//2) - mu2**2
    sigma12 = F.avg_pool2d(img1*img2, window_size, 1, window_size//2) - mu1*mu2

    C1 = 0.01**2
    C2 = 0.03**2

    num = (2*mu1*mu2 + C1) * (2*sigma12 + C2)
    den = (mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2)

    ssim_map = num / den
    return ssim_map.mean().item()

def tensor_to_uint8(img_tensor):
    img = img_tensor.detach().cpu().numpy()
    img = (img * 0.5) + 0.5
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img = np.transpose(img, (1,2,0))
    return img

def vae_loss_fn(recon_x, x, mu, logvar, kl_weight=KL_WEIGHT):
    # Reconstruction loss (L1 instead of MSE for sharper images)
    # Note: x is the ground truth (clean image)
    recon_loss = F.l1_loss(recon_x, x, reduction='mean')
    
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + kl_weight * kld_loss
    return total_loss, recon_loss, kld_loss

# ------------------ Training & Validation ------------------

def train_one_epoch(model, loader, optimizer, device, kl_weight=KL_WEIGHT):
    model.train()
    running_loss = 0.0
    running_recon = 0.0
    running_kld = 0.0

    for inputs, gts, _ in tqdm(loader, desc='train', leave=False):
        inputs = inputs.to(device)
        gts = gts.to(device)

        recon_images, mu, logvar = model(inputs)

        loss, recon_loss, kld_loss = vae_loss_fn(recon_images, gts, mu, logvar, kl_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_recon += recon_loss.item() * inputs.size(0)
        running_kld += kld_loss.item() * inputs.size(0)

    dataset_size = len(loader.dataset)
    return running_loss / dataset_size, running_recon / dataset_size, running_kld / dataset_size

def validate(model, loader, device, sample_indices=None, sample_dir=None):
    model.eval()
    
    running_loss = 0.0
    psnr_list_deb = []
    psnr_list_noisy = []
    ssim_list = []
    dists_list = []

    dists_model = DISTS().to(device).eval()

    saved = 0
    global_idx = 0
    num_samples_to_save = len(sample_indices) if sample_indices else 0

    with torch.no_grad():
        for inputs, gts, fnames in tqdm(loader, desc='val', leave=False):
            inputs = inputs.to(device)
            gts = gts.to(device)

            recon_images, mu, logvar = model(inputs)
            
            # Loss calculation for validation
            loss, _, _ = vae_loss_fn(recon_images, gts, mu, logvar)
            running_loss += loss.item() * inputs.size(0)

            preds_np  = (recon_images.cpu().numpy() + 1.0) / 2.0
            inputs_np = (inputs.cpu().numpy() + 1.0) / 2.0
            gts_np    = (gts.cpu().numpy() + 1.0) / 2.0

            for i in range(preds_np.shape[0]):
                # PSNR noisy
                psnr_noisy = compute_psnr(inputs_np[i].transpose(1,2,0),
                                          gts_np[i].transpose(1,2,0))
                psnr_list_noisy.append(psnr_noisy)

                # PSNR denoised
                psnr = compute_psnr(preds_np[i].transpose(1,2,0),
                                    gts_np[i].transpose(1,2,0))
                psnr_list_deb.append(psnr)

                # DISTS
                pred_t = torch.from_numpy(preds_np[i]).float().to(device).unsqueeze(0)
                gt_t   = torch.from_numpy(gts_np[i]).float().to(device).unsqueeze(0)
                dists_value = dists_model(pred_t, gt_t).item()
                dists_list.append(dists_value)

                # SSIM
                pred_t = torch.tensor(preds_np[i])
                gt_t   = torch.tensor(gts_np[i])
                ssim = compute_ssim(pred_t, gt_t)
                ssim_list.append(ssim)

            if sample_dir is not None and sample_indices:
                for i in range(inputs.size(0)):
                    if global_idx in sample_indices:
                        in_img = tensor_to_uint8(inputs[i])
                        pred_img = tensor_to_uint8(recon_images[i])
                        gt_img = tensor_to_uint8(gts[i])
                        
                        base = os.path.splitext(fnames[i])[0]

                        Image.fromarray(in_img).save(os.path.join(sample_dir, f'{base}_input.png'))
                        Image.fromarray(pred_img).save(os.path.join(sample_dir, f'{base}_pred.png'))
                        Image.fromarray(gt_img).save(os.path.join(sample_dir, f'{base}_gt.png'))
                        
                        saved += 1

                    global_idx += 1
                    if saved >= num_samples_to_save:
                        break

    avg_loss = running_loss / len(loader.dataset)
    avg_psnr_noise = float(np.mean(psnr_list_noisy))
    avg_psnr_deb   = float(np.mean(psnr_list_deb))
    avg_ssim       = float(np.mean(ssim_list))
    avg_dist       = float(np.mean(dists_list))

    return avg_loss, avg_psnr_noise, avg_psnr_deb, avg_ssim, avg_dist

# ------------------ Main runner ------------------

def run_training(train_input, train_gt, val_input, val_gt,
                 img_size=IMG_SIZE, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS,
                 lr=LR, device=DEVICE, checkpoint_dir=CHECKPOINT_DIR, samples_dir=SAMPLES_DIR):

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    train_ds = PairedImageDataset(train_input, train_gt, img_size, augment=True)
    val_ds = PairedImageDataset(val_input, val_gt, img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    sample_indices = pick_random_indices(val_loader, num_samples=8)
    print("Indices d'images sélectionnés pour cette session :", sample_indices)

    model = VAE(img_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_loss': [], 'val_psnr_before': [], 'val_psnr_after': [],'val_ssim': [],'val_dists': []}

    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}/{num_epochs} — training...')

        # KL Annealing: Linear warmup over first 10 epochs
        warmup_epochs = 10
        if epoch <= warmup_epochs:
            current_kl_weight = (epoch / warmup_epochs) * KL_WEIGHT
        else:
            current_kl_weight = KL_WEIGHT

        train_loss, train_recon, train_kld = train_one_epoch(model, train_loader, optimizer, device, kl_weight=current_kl_weight)

        print(f'  Train loss: {train_loss:.6f} (Recon: {train_recon:.6f}, KLD: {train_kld:.6f}, KL Weight: {current_kl_weight:.6f})')

        sample_epoch_dir = os.path.join(samples_dir, f'epoch_{epoch}')
        os.makedirs(sample_epoch_dir, exist_ok=True)
        
        val_loss, val_psnr_before, val_psnr_after, val_ssim, val_dist = validate(
            model,
            val_loader,
            device,
            sample_indices=sample_indices,
            sample_dir=sample_epoch_dir
        )

        print(f'  Val loss: {val_loss:.6f}, Val PSNR before: {val_psnr_before:.3f} dB, Val PSNR after: {val_psnr_after:.3f} dB, Val SSIM : {val_ssim:.4f}, Val DISTS : {val_dist:.5f}')
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_psnr_before'].append(val_psnr_before)
        history['val_psnr_after'].append(val_psnr_after)
        history['val_ssim'].append(val_ssim)
        history['val_dists'].append(val_dist)

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'vae_latest.pt'))
        if epoch % 5 == 0:
             torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'vae_epoch_{epoch}.pt'))

    # Plotting
    plt.figure()
    plt.plot(range(1, len(history['train_loss'])+1), history['train_loss'], label='train_loss')
    plt.plot(range(1, len(history['val_loss'])+1), history['val_loss'], label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, 'loss_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(range(1, len(history['val_psnr_after'])+1), history['val_psnr_after'], label='PSNR')
    plt.xlabel('epoch')
    plt.ylabel('PSNR')
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, 'psnr_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE runner for Denoising.')
    parser.add_argument('--train_input', default=TRAIN_INPUT_DIR)
    parser.add_argument('--train_gt', default=TRAIN_GT_DIR)
    parser.add_argument('--val_input', default=VAL_INPUT_DIR)
    parser.add_argument('--val_gt', default=VAL_GT_DIR)
    parser.add_argument('--img_size', type=int, nargs=2, default=IMG_SIZE)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--device', default=str(DEVICE))
    parser.add_argument('--checkpoint_dir', default=CHECKPOINT_DIR)
    parser.add_argument('--samples_dir', default=SAMPLES_DIR)
    args = parser.parse_args()

    device = torch.device(args.device if args.device != 'cpu' else 'cpu')
    print('Using device:', device)

    for p in [args.train_input, args.train_gt, args.val_input, args.val_gt]:
        if not os.path.exists(p):
            print(f'Warning: Required folder not found: {p}')

    run_training(
        args.train_input, args.train_gt, args.val_input, args.val_gt,
        img_size=tuple(args.img_size), batch_size=args.batch_size, num_epochs=args.epochs,
        lr=args.lr, device=device, checkpoint_dir=args.checkpoint_dir, samples_dir=args.samples_dir
    )
