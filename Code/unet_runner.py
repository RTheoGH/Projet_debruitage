"""
unet_runner.py

Single-file U-Net training & evaluation runner.

Usage (basic):
    python unet_runner.py

Configuration: edit the CONFIG section below (paths, image size, epochs, batch size, ...)
You can also override some values via command-line args (see --help).

Dependencies:
    pip install torch torchvision pillow tqdm numpy matplotlib

What this file contains:
- PairedDataset: loads input / GT images from directories (matching filenames)
- UNet model (DoubleConv, Down, Up, UNet)
- Training loop, validation loop (computes MSE and PSNR)
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

BATCH_SIZE = 32
NUM_EPOCHS = 40
LR = 0.0002
L1_LAMBDA = 100.0
SAVE_EVERY = 1
NUM_WORKERS = 4

CHECKPOINT_DIR = 'model/UNet_runner'
SAMPLES_DIR = 'model/UNet_samples'

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
            raise SystemExit(f'No matching numeric-indexed filenames found between {input_dir} and {gt_dir}.')

        self.pairs = [(in_map[k], gt_map[k]) for k in common]

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

        t_in = TF.normalize(t_in, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        t_gt = TF.normalize(t_gt, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        return t_in, t_gt, in_fname

#-----------------------GAN--------------------------

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key   = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim,      1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query(x).view(B, -1, H * W)
        proj_key   = self.key(x).view(B, -1, H * W)        
        energy     = torch.bmm(proj_query.permute(0, 2, 1), proj_key)
        attention  = F.softmax(energy, dim=-1)
        proj_value = self.value(x).view(B, C, H * W)        

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x


def disc_block(in_c, out_c, stride=2, use_bn=True):
    layers = [nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        channels = in_channels * 2

        self.block1 = disc_block(channels, 64, use_bn=False) 
        self.block2 = disc_block(64, 128)
        self.block3 = disc_block(128, 256)

        self.attn = SelfAttention(256)

        self.block4 = disc_block(256, 512)
        self.block5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)

        h1 = self.block1(inp)
        h2 = self.block2(h1)
        h3 = self.block3(h2)

        h3 = self.attn(h3)

        h4 = self.block4(h3)
        out = self.block5(h4)

        return out          


def hinge_d_loss(real_pred, fake_pred):
    loss_real = torch.relu(1 - real_pred).mean()
    loss_fake = torch.relu(1 + fake_pred).mean()
    return loss_real + loss_fake

def hinge_g_loss(fake_pred):
    return -fake_pred.mean()


# ------------------ Model (UNet) ------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.size() != skip.size():
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNetModel(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        self.inc = DoubleConv(in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(512+512, 256)
        self.up2 = Up(256+256, 128)
        self.up3 = Up(128+128, 64)
        self.up4 = Up(64+64, 64)
        self.outc = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = torch.tanh(x)
        return x

# ------------------ Metrics & helpers ------------------
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

# ------------------ Training & validation ------------------

def train_one_epoch(G, D, loader, opt_G, opt_D, pixel_loss, device):
    G.train()
    D.train()
    running_loss = 0.0

    for inputs, gts, _ in tqdm(loader, desc='train', leave=False):
        inputs = inputs.to(device)
        gts = gts.to(device)

        # -------------------------------
        # 1. Train Discriminator
        # -------------------------------
        preds = G(inputs).detach()

        real_pred = D(inputs, gts)
        fake_pred = D(inputs, preds)

        d_loss = hinge_d_loss(real_pred, fake_pred)

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # -------------------
        # 2. Train Generator 
        # -------------------
        preds = G(inputs)
        fake_pred = D(inputs, preds)

        g_adv = hinge_g_loss(fake_pred)
        g_pixel = pixel_loss(preds, gts)
        g_loss = g_pixel * L1_LAMBDA + g_adv

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        running_loss += g_pixel.item() * inputs.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, loss_fn, device, sample_indices=None, sample_dir=None):
    model.eval()
    running_loss = 0.0
    psnr_list_deb = []
    psnr_list_noisy = []
    ssim_list = []
    dists_list = []
    dists_model = DISTS().to(device)

    saved = 0

    global_idx = 0
    num_samples_to_save = len(sample_indices) if sample_indices else 0


    with torch.no_grad():
        for inputs, gts, fnames in tqdm(loader, desc='val', leave=False):
            inputs = inputs.to(device)
            gts = gts.to(device)
            preds = model(inputs)
            loss = loss_fn(preds, gts)
            running_loss += loss.item() * inputs.size(0)

            preds_np = (preds.cpu().numpy() + 1.0) / 2.0
            inputs_np = (inputs.cpu().numpy() + 1.0) / 2.0
            gts_np = (gts.cpu().numpy() + 1.0) / 2.0
            for i in range(preds_np.shape[0]):
                psnr_noisy = compute_psnr(inputs_np[i].transpose(1,2,0), gts_np[i].transpose(1,2,0))
                psnr_list_noisy.append(psnr_noisy)

                psnr = compute_psnr(preds_np[i].transpose(1,2,0), gts_np[i].transpose(1,2,0))
                psnr_list_deb.append(psnr)


                pred_t = torch.from_numpy(preds_np[i]).float().to(device)
                gt_t   = torch.from_numpy(gts_np[i]).float().to(device)

                pred_t = pred_t.unsqueeze(0)
                gt_t   = gt_t.unsqueeze(0)

                dists_value = dists_model(pred_t, gt_t).item()

                dists_list.append(dists_value)

                pred_t = torch.tensor(preds_np[i])
                gt_t   = torch.tensor(gts_np[i])
                ssim = compute_ssim(pred_t, gt_t)
                ssim_list.append(ssim)
            if sample_dir is not None and sample_indices:
                for i in range(inputs.size(0)):
                    if global_idx in sample_indices:
                        in_img = tensor_to_uint8(inputs[i])
                        pred_img = tensor_to_uint8(preds[i])
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
    avg_psnr_noise = float(np.mean(psnr_list_noisy)) if len(psnr_list_noisy) else 0.0
    avg_psnr_deb = float(np.mean(psnr_list_deb)) if len(psnr_list_deb) else 0.0
    avg_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
    avg_dist = float(np.mean(dists_list)) if len(dists_list) else 0.0

    return avg_loss, avg_psnr_noise, avg_psnr_deb, avg_ssim,avg_dist

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
    # start_idx = 1000
    # num_samples = 8
    # sample_indices = set(range(start_idx, min(start_idx + num_samples, len(val_ds))))
    print("Indices d'images sélectionnés pour cette session :", sample_indices)

    model = UNetModel(in_ch=3, out_ch=3).to(device)
    discriminator = PatchGANDiscriminator().to(device)
    opt_G = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    def lambda_rule(epoch):
        start_decay = num_epochs // 2
        decay_len = num_epochs - start_decay
        if epoch < start_decay:
            return 1.0
        else:
            return 1.0 - float(epoch - start_decay) / float(decay_len + 1)

    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lambda_rule)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda=lambda_rule)

    loss_fn = nn.L1Loss()

    history = {'train_loss': [], 'val_loss': [], 'val_psnr_before': [],'val_psnr_after': []}

    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}/{num_epochs} — training...')

        train_loss = train_one_epoch(model,discriminator, train_loader, opt_G,opt_D, loss_fn, device)

        print(f'  Train loss: {train_loss:.6f}')

        sample_epoch_dir = os.path.join(samples_dir, f'epoch_{epoch}')
        os.makedirs(sample_epoch_dir, exist_ok=True)
        val_loss, val_psnr_before, val_psnr_after, val_ssim, val_dist = validate(
            model,
            val_loader,
            loss_fn,
            device,
            sample_indices=sample_indices,
            sample_dir=sample_epoch_dir
        )

        print(f'  Val loss: {val_loss:.6f},\n Val PSNR before: {val_psnr_before:.3f} dB,\n Val PSNR after: {val_psnr_after:.3f} dB,\n Val SSIM : {val_ssim:.4f}, \n Val DISTS : {val_dist:.5f}')
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_psnr_before'].append(val_psnr_before)
        history['val_psnr_after'].append(val_psnr_after)
        history.setdefault('val_ssim', []).append(val_ssim)

        if epoch % SAVE_EVERY == 0 or epoch == num_epochs:
            ckpt_path = os.path.join(checkpoint_dir, f'unet_epoch_{epoch}.pt')
            torch.save(model.state_dict(), ckpt_path)
            print('  Saved checkpoint', ckpt_path)
        
        sched_G.step()
        sched_D.step()

    plt.figure()
    plt.plot(range(1, len(history['train_loss'])+1), history['train_loss'], label='train_loss')
    plt.plot(range(1, len(history['val_loss'])+1), history['val_loss'], label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plot_path = os.path.join(checkpoint_dir, 'loss_curve.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved loss plot to', plot_path)
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='U-Net runner. Edit top-of-file CONFIG or pass args to override.')
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
            raise SystemExit(f'Required folder not found: {p} — please update the CONFIG variables or pass different args.')

    run_training(
        args.train_input, args.train_gt, args.val_input, args.val_gt,
        img_size=tuple(args.img_size), batch_size=args.batch_size, num_epochs=args.epochs,
        lr=args.lr, device=device, checkpoint_dir=args.checkpoint_dir, samples_dir=args.samples_dir
    )
