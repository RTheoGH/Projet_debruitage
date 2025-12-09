
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from math import log10, sqrt
from DISTS_pt import DISTS

# Import des fonctions depuis votre fichier existant pour garantir la cohérence
from unet_runner import compute_psnr, compute_ssim

def load_image_as_tensor(path, device):
    """Charge une image, la convertit en tenseur [-1, 1] (B, C, H, W)"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image introuvable : {path}")
    
    img = Image.open(path).convert('RGB')
    t = TF.to_tensor(img).to(device) # [0, 1]
    t = (t - 0.5) / 0.5              # [-1, 1]
    return t.unsqueeze(0)

def tensor_to_numpy(t):
    """Convertit un tenseur [-1, 1] en numpy [0, 1] (H, W, C)"""
    t = (t + 1.0) / 2.0
    t = torch.clamp(t, 0, 1)
    return t.squeeze(0).permute(1, 2, 0).cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Calculer PSNR, SSIM et DISTS entre deux images.")
    parser.add_argument("ref_path", type=str, help="Chemin vers l'image de référence (Vérité Terrain)")
    parser.add_argument("dist_path", type=str, help="Chemin vers l'image à évaluer (Débruitée/Générée)")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")

    # 1. Chargement
    try:
        t_ref = load_image_as_tensor(args.ref_path, device)
        t_dist = load_image_as_tensor(args.dist_path, device)
    except Exception as e:
        print(f"Erreur lors du chargement des images : {e}")
        return

    # 2. Vérification des tailles
    if t_ref.shape != t_dist.shape:
        print(f"ATTENTION : Les tailles diffèrent.")
        print(f"Ref : {t_ref.shape}, Dist : {t_dist.shape}")
        print("Redimensionnement de l'image évaluée vers la taille de référence...")
        t_dist = F.interpolate(t_dist, size=t_ref.shape[2:], mode='bilinear', align_corners=False)

    # 3. Préparation des données pour chaque métrique
    
    # Pour PSNR (Numpy [0, 1])
    np_ref = tensor_to_numpy(t_ref)
    np_dist = tensor_to_numpy(t_dist)
    
    # Pour SSIM (Tenseur [C, H, W] sans batch, car compute_ssim ajoute unsqueeze)
    t_ref_sq = t_ref.squeeze(0)
    t_dist_sq = t_dist.squeeze(0)

    # 4. Calculs
    print("-" * 30)
    print("Calcul des métriques...")
    
    # PSNR
    psnr_val = compute_psnr(np_ref, np_dist, max_val=1.0)
    
    # SSIM
    ssim_val = compute_ssim(t_dist_sq, t_ref_sq)
    
    # DISTS
    dists_model = DISTS().to(device)
    dists_val = dists_model(t_dist, t_ref).item()

    # 5. Affichage
    print("-" * 30)
    print(f"Reference : {args.ref_path}")
    print(f"Evaluated : {args.dist_path}")
    print("-" * 30)
    print(f"PSNR  : {psnr_val:.2f} dB  (Plus haut est mieux)")
    print(f"SSIM  : {ssim_val:.4f}     (Plus proche de 1 est mieux)")
    print(f"DISTS : {dists_val:.4f}     (Plus proche de 0 est mieux)")
    print("-" * 30)

import os

if __name__ == "__main__":
    main()
