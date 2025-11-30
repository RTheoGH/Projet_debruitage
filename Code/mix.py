import cv2
import numpy as np
import os
import argparse

def gaussNoise(image, sigma=25):
    img = image.astype(np.float32)
    gauss = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def periodicNoise(image, amplitude=25, freq_x=0.05, freq_y=0.08):
    img = image.astype(np.float32)
    h, w = img.shape[:2]

    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)

    noise = amplitude * np.sin(2 * np.pi * (freq_x * X + freq_y * Y))

    if img.ndim == 3:
        noise = noise[:, :, np.newaxis]

    noisy = img + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def noise_poisson(image):
    img = image.astype(np.float32)
    noisy = np.random.poisson(img / 255.0 * 100) / 100.0 * 255.0
    return np.clip(noisy, 0, 255).astype(np.uint8)


def salt_and_pepper(image, salt_ratio=0.05, pepper_ratio=0.05):
    noisy = image.copy()
    if noisy.ndim == 2:
        row, col = noisy.shape
        salt = np.random.rand(row, col) < salt_ratio
        pepper = np.random.rand(row, col) < pepper_ratio
        noisy[salt] = 255
        noisy[pepper] = 0
    else:
        row, col, ch = noisy.shape
        salt = np.random.rand(row, col) < salt_ratio
        pepper = np.random.rand(row, col) < pepper_ratio
        for c in range(ch):
            noisy[..., c][salt] = 255
            noisy[..., c][pepper] = 0
    return noisy

NOISES = {
    "gaussian": gaussNoise,
    "periodic": periodicNoise,
    "poisson": noise_poisson,
    "saltpepper": salt_and_pepper
}

def mix_noises(img, noise1, noise2, alpha):
    n1 = NOISES[noise1](img).astype(np.float32)
    n2 = NOISES[noise2](img).astype(np.float32)
    mixed = alpha * n1 + (1 - alpha) * n2
    return np.clip(mixed, 0, 255).astype(np.uint8)

def process(noise1, noise2, alpha):
    base = os.path.join('..', 'allImages')

    for split in ('train', 'validation'):
        src = os.path.join(base, split, 'truth')
        out_dir = os.path.join(base, split, 'noised', f"{noise1}_{noise2}")
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.isdir(src):
            continue

        allowed_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        files = sorted(f for f in os.listdir(src) if f.lower().endswith(allowed_exts))

        for f in files:
            src_path = os.path.join(src, f)
            img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)

            if img is None:
                print(f"Impossible de lire {src_path}")
                continue

            mixed = mix_noises(img, noise1, noise2, alpha)

            out_path = os.path.join(out_dir, f)
            if not cv2.imwrite(out_path, mixed):
                print(f"Impossible d'écrire {out_path}")

        print(f"{split} terminé - {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("noise1", choices=NOISES.keys(), help="Premier bruit")
    parser.add_argument("noise2", choices=NOISES.keys(), help="Deuxième bruit")
    parser.add_argument("--alpha", type=float, default=0.5, help="Poids du premier bruit (0 à 1)")

    args = parser.parse_args()

    process(args.noise1, args.noise2, args.alpha)
