import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import sys, os

# même booléen que dans unet_runner.py
RESIDUAL_LEARNING = True


# ------------------ U-Net identique au runner ------------------
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

    def forward(self, x): return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x): return self.net(x)


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
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        x_input = x
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

        if RESIDUAL_LEARNING:
            return torch.tanh(x + x_input)
        else:
            return torch.tanh(x)


# ------------------ Load model ------------------
def load_model(model_path, device="cpu"):
    model = UNetModel(in_ch=3, out_ch=3).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ------------------ Denoise image ------------------
def denoise_image(model, input_image, device="cpu"):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   # même normalisation que l’entraînement
    ])

    x = transform(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        y = model(x)

    # dénormalisation depuis [-1,1] vers [0,1]
    y = (y * 0.5 + 0.5).clamp(0,1).squeeze().cpu()

    return T.ToPILImage()(y)


# ------------------ Main ------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python debruitage.py chemin_image.png")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print("Erreur : l'image n'existe pas :", image_path)
        sys.exit(1)

    MODEL_PATH = "../models/gauss_m/UNet_runner/model_epoch_20.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(MODEL_PATH, device)

    img = Image.open(image_path).convert("RGB")
    denoised = denoise_image(model, img, device)

    output_path = "../testImg/denoised.png"
    denoised.save(output_path)
    print("Image sauvegardée :", output_path)
