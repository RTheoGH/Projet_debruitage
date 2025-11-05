import numpy as np
import random
import cv2
import os

def noise_poisson(image):
    noisy = np.random.poisson(image / 255.0 * 100) / 100 * 255
    return noisy

path = '../images'
dirs = os.listdir(path)

for file in dirs:
    image = cv2.imread(path+'/'+file, cv2.IMREAD_GRAYSCALE)
    noisy_image = noise_poisson(image)
    cv2.imwrite('../poisson/'+file, noisy_image)