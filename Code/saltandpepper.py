import numpy as np
import cv2
import os

def add_salt_and_pepper(image, salt_ratio=0.05, pepper_ratio=0.05):
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


def process():
    base = os.path.join('..', 'allImages')
    for split in ('train', 'validation'):
        src = os.path.join(base, split, 'truth')
        out_dir = os.path.join(base, split, 'noised', 'saltandpepper')
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

            noisy = add_salt_and_pepper(img)

            out_name = f
            out_path = os.path.join(out_dir, out_name)
            written = cv2.imwrite(out_path, noisy)
            if not written:
                print(f"Error: impossible d'écrire '{out_path}'")
            else:
                print(f"Sauvegardé: {out_path}")


if __name__ == '__main__':
    process()