import numpy as np
import cv2
import os


def noise_poisson(image):
    img = image.astype(np.float32)
    noisy = np.random.poisson(img / 255.0 * 100) / 100.0 * 255.0
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def process():
    base = os.path.join('..', 'allImages')
    for split in ('train', 'validation'):
        src = os.path.join(base, split, 'truth')
        out_dir = os.path.join(base, split, 'noised', 'poisson')
        os.makedirs(out_dir, exist_ok=True)
        if not os.path.isdir(src):
            continue
        files = sorted(f for f in os.listdir(src) if f.lower().endswith('.png'))
        idx = 1
        for f in files:
            img = cv2.imread(os.path.join(src, f), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            noisy = noise_poisson(img)
            out_name = f'poisson{idx}.png'
            cv2.imwrite(os.path.join(out_dir, out_name), noisy)
            idx += 1


if __name__ == '__main__':
    process()