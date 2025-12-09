import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import log10, sqrt
from unet_runner import UNetModel, RESIDUAL_LEARNING, DISTS, compute_ssim

# IMG_PATH_GT = '../allImages256/validation/truth/image02311.jpg'    
# IMG_PATH_NOISY = '../allImages256/validation/noised/gaussian/image02311.jpg' 

IMG_PATH_GT = '../testImg/noise.jpg'    
IMG_PATH_NOISY = '../testImg/noise.jpg' 

MODEL_PATH = 'gauss2_m/model_final.pt'
OUTPUT_DIR = 'resultats_subdivision'

PATCH_SIZE = 128
STRIDE = 64 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tensor_to_pil(t):
    """Convertit un tenseur [-1, 1] en image PIL"""
    t = (t + 1.0) / 2.0
    t = torch.clamp(t, 0, 1)
    t = t.squeeze(0).cpu()
    return TF.to_pil_image(t)

def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * log10(255.0 / sqrt(mse))

def get_patches(img_tensor, patch_size=128, stride=64):
    """Découpe l'image en patchs avec recouvrement"""
    img = img_tensor.squeeze(0)
    _, h, w = img.shape
    patches = []
    coords = []
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coords.append((y, x))
            
    if h % stride != 0:
        y = h - patch_size
        for x in range(0, w - patch_size + 1, stride):
            patch = img[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coords.append((y, x))
            
    if w % stride != 0:
        x = w - patch_size
        for y in range(0, h - patch_size + 1, stride):
            patch = img[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coords.append((y, x))

    return torch.stack(patches), coords

def reconstruct_from_patches(patches, coords, full_shape, patch_size=128):
    """Reconstruit l'image en faisant la moyenne des recouvrements"""
    _, c, h, w = full_shape
    result = torch.zeros(full_shape).to(patches.device)
    weights = torch.zeros(full_shape).to(patches.device)
    
    patch_weight = torch.ones((1, c, patch_size, patch_size)).to(patches.device)

    for i, (y, x) in enumerate(coords):
        result[:, :, y:y+patch_size, x:x+patch_size] += patches[i]
        weights[:, :, y:y+patch_size, x:x+patch_size] += patch_weight
        
    weights[weights == 0] = 1.0
    return result / weights


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'patches_noisy'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'patches_denoised'), exist_ok=True)

    print(f"Chargement du modèle depuis {MODEL_PATH}...")
    model = UNetModel(in_ch=3, out_ch=3).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print(f"ATTENTION: Modèle non trouvé à {MODEL_PATH}. Utilisation de poids aléatoires pour le test.")
    
    model.eval()

    print("Chargement des images...")
    if not os.path.exists(IMG_PATH_GT):
        print(f"Erreur : Image {IMG_PATH_GT} introuvable.")
        print("Création d'une image synthétique pour le test...")
        img_gt_pil = Image.new('RGB', (512, 512), color='red')
    else:
        img_gt_pil = Image.open(IMG_PATH_GT).convert('RGB')
    
    if os.path.exists(IMG_PATH_NOISY):
        img_noisy_pil = Image.open(IMG_PATH_NOISY).convert('RGB')
    else:
        print("Image bruitée non trouvée, génération de bruit gaussien...")
        img_array = np.array(img_gt_pil) / 255.0
        noise = np.random.normal(0, 0.1, img_array.shape)
        img_noisy_array = np.clip(img_array + noise, 0, 1)
        img_noisy_pil = Image.fromarray((img_noisy_array * 255).astype(np.uint8))

    t_gt = TF.to_tensor(img_gt_pil).unsqueeze(0).to(DEVICE)
    t_noisy = TF.to_tensor(img_noisy_pil).unsqueeze(0).to(DEVICE)

    t_gt_norm = (t_gt - 0.5) / 0.5
    t_noisy_norm = (t_noisy - 0.5) / 0.5

    print(f"Taille de l'image : {t_gt.shape}")

    print("\n--- Subdivision (Tiling) ---")
    
    patches_noisy, coords = get_patches(t_noisy_norm, PATCH_SIZE, STRIDE)
    print(f"Image découpée en {len(patches_noisy)} patchs.")

    patches_denoised = []
    
    batch_size = 8
    with torch.no_grad():
        for i in tqdm(range(0, len(patches_noisy), batch_size), desc="Débruitage des patchs"):
            batch = patches_noisy[i:i+batch_size]
            pred = model(batch)
            patches_denoised.append(pred)
            
            for j, p_out in enumerate(pred):
                idx = i + j
                p_in_pil = tensor_to_pil(batch[j])
                p_in_pil.save(os.path.join(OUTPUT_DIR, 'patches_noisy', f'patch_{idx:03d}.png'))
                p_out_pil = tensor_to_pil(p_out)
                p_out_pil.save(os.path.join(OUTPUT_DIR, 'patches_denoised', f'patch_{idx:03d}.png'))

    patches_denoised = torch.cat(patches_denoised)

    t_reconstructed_norm = reconstruct_from_patches(patches_denoised, coords, t_noisy.shape, PATCH_SIZE)
    
    img_tiled = tensor_to_pil(t_reconstructed_norm)
    img_tiled.save(os.path.join(OUTPUT_DIR, 'resultat_tiling.png'))


    print("\n--- Calcul des métriques ---")
    
    np_gt = np.array(img_gt_pil)
    np_noisy = np.array(img_noisy_pil)
    np_tiled = np.array(img_tiled)

    psnr_noisy = compute_psnr(np_gt, np_noisy)
    psnr_tiled = compute_psnr(np_gt, np_tiled)

    print(f"PSNR Input Bruité : {psnr_noisy:.2f} dB")
    print(f"PSNR Méthode Tiling : {psnr_tiled:.2f} dB ")

    ssim_tiled = compute_ssim(t_reconstructed_norm.squeeze(0), t_gt_norm.squeeze(0))
    print(f"SSIM Tiling : {ssim_tiled:.4f}")

    dists_model = DISTS().to(DEVICE)
    d_tiled = dists_model(t_reconstructed_norm, t_gt_norm).item()
    
    print(f"DISTS Tiling : {d_tiled:.4f}")

if __name__ == "__main__":
    main()
