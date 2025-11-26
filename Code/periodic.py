import cv2
import numpy as np
import os

def periodicNoise(image, amplitude=50, freq_x=0.1, freq_y=0.1):
    img = image.astype(np.float32)
    h, w = img.shape[:2]

    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)

    noise = amplitude * np.sin(2 * np.pi * (freq_x * X + freq_y * Y))

    if img.ndim == 3:
        noise = noise[:, :, np.newaxis]

    noisy = img + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def process():
    base = os.path.join('..', 'allImages')
    for split in ('train', 'validation'):
        src = os.path.join(base, split, 'truth')
        out_dir = os.path.join(base, split, 'noised', 'periodic')
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.isdir(src):
            continue

        allowed_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        files = sorted(f for f in os.listdir(src) if f.lower().endswith(allowed_exts))

        for f in files:
            src_path = os.path.join(src, f)
            img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Warning: impossible de lire '{src_path}', skipping")
                continue

            noisy = periodicNoise(img)

            out_path = os.path.join(out_dir, f)
            written = cv2.imwrite(out_path, noisy)
            if not written:
                print(f"Error: impossible d'écrire '{out_path}'")

if __name__ == "__main__":
    process()