import cv2
import numpy as np
import os


def gaussNoise(image, sigma=25):
    img = image.astype(np.float32)
    gauss = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def process():
    base = os.path.join('..', 'allImages')
    for split in ('train', 'validation'):
        src = os.path.join(base, split, 'truth')
        out_dir = os.path.join(base, split, 'noised', 'gaussian')
        os.makedirs(out_dir, exist_ok=True)
        if not os.path.isdir(src):
            continue
        files = sorted(f for f in os.listdir(src) if f.lower().endswith('.png'))
        idx = 1
        for f in files:
            img = cv2.imread(os.path.join(src, f), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            noisy = gaussNoise(img)
            out_name = f'gauss{idx}.png'
            cv2.imwrite(os.path.join(out_dir, out_name), noisy)
            idx += 1


if __name__ == "__main__":
    process()