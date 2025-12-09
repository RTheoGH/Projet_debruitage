
import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from math import log10, sqrt
from tqdm import tqdm
from unet_runner import UNetModel, DISTS, compute_ssim

IMG_PATH_GT = '../allImages/validation/truth/image04390.jpg'
IMG_PATH_NOISY = '../allImages/validation/noised/gaussian/image04390.jpg'

MODEL_PATH = 'gauss2_m/model_final.pt'
OUTPUT_DIR = 'resultats_zoom_subdivision'

INPUT_SIZE = 128        
SUB_PATCH_SIZE = 64     
STRIDE = 32
MODEL_INPUT_SIZE = 128  

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tensor_to_pil(t):
    t = (t + 1.0) / 2.0
    t = torch.clamp(t, 0, 1)
    t = t.squeeze(0).cpu()
    return TF.to_pil_image(t)

def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * log10(255.0 / sqrt(mse))

def get_patches(img_tensor, patch_size, stride):
    """Découpe l'image en patchs"""
    img = img_tensor.squeeze(0)
    _, h, w = img.shape
    patches = []
    coords = []
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coords.append((y, x))
    return torch.stack(patches), coords

def reconstruct_from_patches(patches, coords, full_shape, patch_size):
    """Reconstruit l'image"""
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
    os.makedirs(os.path.join(OUTPUT_DIR, 'details'), exist_ok=True)

    print(f"Chargement du modèle {MODEL_PATH}...")
    model = UNetModel(in_ch=3, out_ch=3).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print("Modèle introuvable, utilisation poids aléatoires.")
    model.eval()

    print("Préparation des images...")
    if os.path.exists(IMG_PATH_GT):
        img_gt = Image.open(IMG_PATH_GT).convert('RGB').resize((INPUT_SIZE, INPUT_SIZE))
    else:
        img_gt = Image.new('RGB', (INPUT_SIZE, INPUT_SIZE), color='green')
    
    img_array = np.array(img_gt) / 255.0
    noise = np.random.normal(0, 0.1, img_array.shape)
    img_noisy_array = np.clip(img_array + noise, 0, 1)
    img_noisy = Image.fromarray((img_noisy_array * 255).astype(np.uint8))

    img_gt.save(os.path.join(OUTPUT_DIR, 'original_gt.png'))
    img_noisy.save(os.path.join(OUTPUT_DIR, 'original_noisy.png'))

    t_noisy = TF.to_tensor(img_noisy).unsqueeze(0).to(DEVICE)
    t_noisy_norm = (t_noisy - 0.5) / 0.5
    t_gt = TF.to_tensor(img_gt).unsqueeze(0).to(DEVICE)
    t_gt_norm = (t_gt - 0.5) / 0.5

    print("\n--- Méthode 1 : Standard (Direct) ---")
    with torch.no_grad():
        out_standard = model(t_noisy_norm)
    
    img_standard = tensor_to_pil(out_standard)
    img_standard.save(os.path.join(OUTPUT_DIR, 'resultat_standard.png'))

    print(f"\n--- Méthode 2 : Zoom & Subdivision ({SUB_PATCH_SIZE}x{SUB_PATCH_SIZE} -> {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE}) ---")
    
    patches, coords = get_patches(t_noisy_norm, SUB_PATCH_SIZE, STRIDE)
    print(f"Nombre de patchs : {len(patches)}")

    processed_patches = []
    
    with torch.no_grad():
        for i, patch in enumerate(tqdm(patches)):
            patch_input = patch.unsqueeze(0)
            patch_zoomed = F.interpolate(patch_input, size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), mode='bilinear', align_corners=False)

            patch_out_zoomed = model(patch_zoomed)
            
            patch_out_small = F.interpolate(patch_out_zoomed, size=(SUB_PATCH_SIZE, SUB_PATCH_SIZE), mode='bilinear', align_corners=False)
            
            processed_patches.append(patch_out_small.squeeze(0))

            if i < 5:
                tensor_to_pil(patch_input).save(os.path.join(OUTPUT_DIR, 'details', f'p{i}_input_small.png'))
                tensor_to_pil(patch_zoomed).save(os.path.join(OUTPUT_DIR, 'details', f'p{i}_zoomed_input.png'))
                tensor_to_pil(patch_out_zoomed).save(os.path.join(OUTPUT_DIR, 'details', f'p{i}_zoomed_output.png'))
                tensor_to_pil(patch_out_small).save(os.path.join(OUTPUT_DIR, 'details', f'p{i}_output_small.png'))

    processed_patches = torch.stack(processed_patches)
    
    t_reconstructed = reconstruct_from_patches(processed_patches, coords, t_noisy.shape, SUB_PATCH_SIZE)
    img_zoom = tensor_to_pil(t_reconstructed)
    img_zoom.save(os.path.join(OUTPUT_DIR, 'resultat_zoom.png'))


    print("\n--- Résultats ---")
    np_gt = np.array(img_gt)
    np_std = np.array(img_standard)
    np_zoom = np.array(img_zoom)

    psnr_std = compute_psnr(np_gt, np_std)
    psnr_zoom = compute_psnr(np_gt, np_zoom)

    ssim_std = compute_ssim(out_standard.squeeze(0), t_gt_norm.squeeze(0))
    ssim_zoom = compute_ssim(t_reconstructed.squeeze(0), t_gt_norm.squeeze(0))

    dists_model = DISTS().to(DEVICE)
    d_std = dists_model(out_standard, t_gt_norm).item()
    d_zoom = dists_model(t_reconstructed, t_gt_norm).item()

    print(f"PSNR Standard : {psnr_std:.2f} dB")
    print(f"PSNR Zoom     : {psnr_zoom:.2f} dB")
    print("-" * 20)
    print(f"SSIM Standard : {ssim_std:.4f}")
    print(f"SSIM Zoom     : {ssim_zoom:.4f}")
    print("-" * 20)
    print(f"DISTS Standard : {d_std:.4f}")
    print(f"DISTS Zoom     : {d_zoom:.4f}")

    if psnr_zoom > psnr_std:
        print("\n=> La méthode ZOOM semble meilleure !")
    else:
        print("\n=> La méthode STANDARD est meilleure (probablement à cause de l'échelle du bruit).")

if __name__ == "__main__":
    main()
