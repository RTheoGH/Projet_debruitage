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
        
        # Correction: Utiliser la connexion résiduelle et Tanh comme dans l'entraînement
        out = x + x_input
        out = torch.tanh(out)
        return out

# ------------------ Load model ------------------
def load_model(model_path, device="cpu"):
    model = UNetModel(in_ch=3, out_ch=3).to(device)
    # map_location ensures we can load a cuda model on cpu if needed
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ------------------ Denoise image ------------------
def denoise_image(model, input_image, device="cpu"):
    # Correction: Normalisation [-1, 1] comme à l'entraînement
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    x = transform(input_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        y = model(x)
    
    # Correction: Dénormalisation [-1, 1] -> [0, 1]
    y = (y.squeeze() * 0.5) + 0.5
    y = y.clamp(0, 1).cpu()
    
    return T.ToPILImage()(y)


# ------------------ Main ------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python debruitage.py chemin_image.png [chemin_modele]")
        sys.exit(1)

    image_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        MODEL_PATH = sys.argv[2]
    else:
        MODEL_PATH = "./gauss2_m/model_final.pt" 
        if not os.path.exists(MODEL_PATH):
             MODEL_PATH = "./weights.pt" # Fallback

    if not os.path.exists(image_path):
        print("Erreur : l'image n'existe pas :", image_path)
        sys.exit(1)
        
    if not os.path.exists(MODEL_PATH):
        print(f"Attention: Le modèle par défaut n'a pas été trouvé à {MODEL_PATH}.")
        print("Veuillez spécifier le chemin du modèle en 2ème argument.")
    
    print(f"Chargement du modèle depuis : {MODEL_PATH}")

    MODEL_PATH = "../models/gauss_m/UNet_runner/model_epoch_20.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(MODEL_PATH, device)

    img = Image.open(image_path).convert("RGB")
    denoised = denoise_image(model, img, device)

    output_path = "../testImg/denoised.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    denoised.save(output_path)
    print(f"Image débruitée sauvegardée sous : {output_path}")
