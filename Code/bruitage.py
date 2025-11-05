#!/usr/bin/env python3
import cv2
import numpy as np

import pickle


def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, sigma: float = 25.0, clip: bool = True, seed: int | None = None) -> np.ndarray:
    """
    Add Gaussian noise to an image.

    Inputs:
    - image: np.ndarray, image in HxW or HxWxC format. dtype uint8 or float.
    - mean: mean of gaussian noise
    - sigma: std-dev of gaussian noise
    - clip: if True, clip values to valid range and return uint8
    - seed: optional random seed for reproducibility

    Returns:
    - noisy image as np.ndarray (same shape). If clip=True returns dtype uint8 in [0,255].
    """
    if seed is not None:
        np.random.seed(seed)

    orig_dtype = image.dtype
    img_float = image.astype(np.float32)

    noise = np.random.normal(loc=mean, scale=sigma, size=img_float.shape).astype(np.float32)

    noisy = img_float + noise

    if clip:
        if np.issubdtype(orig_dtype, np.integer):
            info = np.iinfo(orig_dtype)
            noisy = np.clip(noisy, info.min, info.max)
            return noisy.astype(orig_dtype)
        else:
            noisy = np.clip(noisy, 0.0, 1.0)
            return noisy.astype(orig_dtype)
    else:
        return noisy

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def main():

    dic1 = unpickle('../cifar-10-batches-py/data_batch_1')
    dic2 = unpickle('../cifar-10-batches-py/data_batch_1')
    dic3 = unpickle('../cifar-10-batches-py/data_batch_1')
    dic4 = unpickle('../cifar-10-batches-py/data_batch_1')
    dic5 = unpickle('../cifar-10-batches-py/data_batch_1')

    dic1b = dic1[b'data']
    dic2b = dic2[b'data']
    dic3b = dic3[b'data']
    dic4b = dic4[b'data']
    dic5b = dic5[b'data']
    dic1b = dic1b.reshape(10000,3,32,32).transpose(0,2,3,1)
    dic2b = dic2b.reshape(10000,3,32,32).transpose(0,2,3,1)
    dic3b = dic3b.reshape(10000,3,32,32).transpose(0,2,3,1)
    dic4b = dic4b.reshape(10000,3,32,32).transpose(0,2,3,1)
    dic5b = dic5b.reshape(10000,3,32,32).transpose(0,2,3,1)

    for i in range(dic1b.shape[0]):
        noisy_img = add_gaussian_noise(dic1b[i], mean=0, sigma=25, clip=True, seed=None)

        cv2.imwrite(f'../images/ori1-{i}.png', dic1b[i])

    for i in range(dic2b.shape[0]):
        noisy_img = add_gaussian_noise(dic2b[i], mean=0, sigma=25, clip=True, seed=None)
        cv2.imwrite(f'../images/ori2-{i}.png', dic2b[i])

    for i in range(dic3b.shape[0]):
        noisy_img = add_gaussian_noise(dic3b[i], mean=0, sigma=25, clip=True, seed=None)
        cv2.imwrite(f'../images/ori3-{i}.png', dic3b[i])

    for i in range(dic4b.shape[0]):
        noisy_img = add_gaussian_noise(dic4b[i], mean=0, sigma=25, clip=True, seed=None)
        cv2.imwrite(f'../images/ori4-{i}.png', dic4b[i])

    for i in range(dic5b.shape[0]):
        noisy_img = add_gaussian_noise(dic5b[i], mean=0, sigma=25, clip=True, seed=None)
        cv2.imwrite(f'../images/ori5-{i}.png', dic5b[i])

if __name__ == "__main__":
    main()