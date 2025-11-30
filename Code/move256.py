import cv2
import numpy as np
import pickle
import os
import shutil


def copy_all(src_dirs, dst_dir, start_idx=0):
    counter = start_idx

    for src in src_dirs:
        if not os.path.isdir(src):
            print(f"[WARN] Le dossier {src} n'existe pas.")
            continue

        for file in os.listdir(src):
            src_file = os.path.join(src, file)

            if os.path.isfile(src_file):
                ext = os.path.splitext(file)[1]

                new_name = f"image{counter:05d}{ext}"
                dst_file = os.path.join(dst_dir, new_name)

                shutil.copy2(src_file, dst_file)

                counter += 1

    return counter


def main():
    out_train_truth = os.path.join('..', 'allImages', 'train', 'truth')
    out_val_truth = os.path.join('..', 'allImages', 'validation', 'truth')
    os.makedirs(out_train_truth, exist_ok=True)
    os.makedirs(out_val_truth, exist_ok=True)

    in_train_truth = [
        "../data/images256-256/chat/data/train/egyptian_cat/",
        "../data/images256-256/hare/data/train/hare/",
        "../data/images256-256/weasel/data/train/weasel/"
    ]

    in_val_truth = [
        "../data/images256-256/chat/data/val/egyptian_cat/",
        "../data/images256-256/hare/data/val/hare/",
        "../data/images256-256/weasel/data/val/weasel/"
    ]

    print("\n--- Copie TRAIN ---")
    last_idx = copy_all(in_train_truth, out_train_truth, start_idx=0)

    print("\n--- Copie VAL ---")
    copy_all(in_val_truth, out_val_truth, start_idx=last_idx)


if __name__ == "__main__":
    print("Working directory:", os.getcwd())
    main()
