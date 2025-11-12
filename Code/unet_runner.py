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
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ------------------ CONFIG------------------
TRAIN_INPUT_DIR = '../allImages/train/noised/gaussian'  
TRAIN_GT_DIR =    '../allImages/train/truth' 
VAL_INPUT_DIR =   '../allImages/validation/noised/gaussian'  
VAL_GT_DIR =      '../allImages/validation/truth'
TEST_INPUT_DIR =  '../allImages/validation/noised/gaussian/test'
TEST_GT_DIR =     '../allImages/validation/truth/test'

IMG_SIZE = (128, 128)

BATCH_SIZE = 3
NUM_EPOCHS = 3
LR = 1e-3
SAVE_EVERY = 1
NUM_WORKERS = 4

CHECKPOINT_DIR = 'model/UNet_runner'
SAMPLES_DIR = 'model/UNet_samples'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PairedImageDataset(Dataset):
    """Load pairs of images (input, gt) from two folders. Filenames must match."""
    def __init__(self, input_dir, gt_dir, img_size=(32,32), transform=None):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
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
        self.img_size = img_size
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),          
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        in_fname, gt_fname = self.pairs[idx]
        p_in = os.path.join(self.input_dir, in_fname)
        p_gt = os.path.join(self.gt_dir, gt_fname)
        img_in = Image.open(p_in).convert('RGB')
        img_gt = Image.open(p_gt).convert('RGB')
        t_in = self.transform(img_in)
        t_gt = self.transform(img_gt)
        return t_in, t_gt, in_fname


# ------------------ Model (UNet) ------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
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
        x = torch.sigmoid(x)
        return x

# ------------------ Metrics & helpers ------------------
def compute_psnr(img1, img2, max_val=1.0):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * log10(max_val / sqrt(mse))


def tensor_to_uint8(img_tensor):
    img = img_tensor.detach().cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img = np.transpose(img, (1,2,0))
    return img

# ------------------ Training & validation ------------------

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm(loader, desc='train', leave=False):
        inputs, gts, _ = batch
        inputs = inputs.to(device)
        gts = gts.to(device)
        preds = model(inputs)
        loss = loss_fn(preds, gts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    avg_loss = running_loss / len(loader.dataset)
    return avg_loss


def validate(model, loader, loss_fn, device, num_samples_to_save=4, sample_dir=None):
    model.eval()
    running_loss = 0.0
    psnr_list = []
    saved = 0
    with torch.no_grad():
        for inputs, gts, fnames in tqdm(loader, desc='val', leave=False):
            inputs = inputs.to(device)
            gts = gts.to(device)
            preds = model(inputs)
            loss = loss_fn(preds, gts)
            running_loss += loss.item() * inputs.size(0)
            preds_np = preds.cpu().numpy()
            gts_np = gts.cpu().numpy()
            for i in range(preds_np.shape[0]):
                psnr = compute_psnr(preds_np[i].transpose(1,2,0), gts_np[i].transpose(1,2,0))
                psnr_list.append(psnr)
            if sample_dir is not None and saved < num_samples_to_save:
                for i in range(min(inputs.size(0), num_samples_to_save - saved)):
                    in_img = tensor_to_uint8(inputs[i])
                    pred_img = tensor_to_uint8(preds[i])
                    gt_img = tensor_to_uint8(gts[i])
                    base = os.path.splitext(fnames[i])[0]
                    Image.fromarray(in_img).save(os.path.join(sample_dir, f'{base}_input.png'))
                    Image.fromarray(pred_img).save(os.path.join(sample_dir, f'{base}_pred.png'))
                    Image.fromarray(gt_img).save(os.path.join(sample_dir, f'{base}_gt.png'))
                    saved += 1
                    if saved >= num_samples_to_save:
                        break
    avg_loss = running_loss / len(loader.dataset)
    avg_psnr = float(np.mean(psnr_list)) if len(psnr_list) else 0.0
    return avg_loss, avg_psnr

# ------------------ Main runner ------------------

def run_training(train_input, train_gt, val_input, val_gt,
                 img_size=IMG_SIZE, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS,
                 lr=LR, device=DEVICE, checkpoint_dir=CHECKPOINT_DIR, samples_dir=SAMPLES_DIR):

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    train_ds = PairedImageDataset(train_input, train_gt, img_size)
    val_ds = PairedImageDataset(val_input, val_gt, img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    model = UNetModel(in_ch=3, out_ch=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': [], 'val_psnr': []}

    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}/{num_epochs} — training...')
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f'  Train loss: {train_loss:.6f}')
        # validation
        sample_epoch_dir = os.path.join(samples_dir, f'epoch_{epoch}')
        os.makedirs(sample_epoch_dir, exist_ok=True)
        val_loss, val_psnr = validate(model, val_loader, loss_fn, device, num_samples_to_save=8, sample_dir=sample_epoch_dir)
        print(f'  Val loss: {val_loss:.6f}, Val PSNR: {val_psnr:.3f} dB')
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_psnr'].append(val_psnr)
        # save checkpoint
        if epoch % SAVE_EVERY == 0 or epoch == num_epochs:
            ckpt_path = os.path.join(checkpoint_dir, f'unet_epoch_{epoch}.pt')
            torch.save(model.state_dict(), ckpt_path)
            print('  Saved checkpoint', ckpt_path)

    # final plot
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
