import numpy as np
import random
import cv2
import os

def add_salt_and_pepper(image, salt_ratio=0.05,pepper_ratio=0.05):
    """
    Adds salt and pepper noise to an image.
    """
    row, col = image.shape
    salt = np.random.rand(row, col) < salt_ratio
    pepper = np.random.rand(row, col) < pepper_ratio
    noisy_image = np.copy(image)
    noisy_image[salt] = 1
    noisy_image[pepper] = 0
    return noisy_image

path = '../images'
dirs = os.listdir(path)

for file in dirs:
    image = cv2.imread(path+'/'+file, cv2.IMREAD_GRAYSCALE)
    noisy_image = add_salt_and_pepper(image)
    cv2.imwrite('../saltandpepper/'+file, noisy_image)