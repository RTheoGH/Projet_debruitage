import cv2
import numpy as np
import os

def gaussNoise(image):
    row,col,ch= image.shape
    mean = 0
    sigma = 25
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy

def main():

    for filename in os.listdir('../images'):
        if filename.endswith('.png'):
            img = cv2.imread(os.path.join('../images', filename))
            noisy_img = gaussNoise(img)
            cv2.imwrite('../gaussian/' + filename, noisy_img)

if __name__ == "__main__":
    main()