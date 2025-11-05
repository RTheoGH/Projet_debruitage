
import cv2
import numpy as np
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def main():

    dic1 = unpickle('../cifar-10-batches-py/data_batch_1')
    dic2 = unpickle('../cifar-10-batches-py/data_batch_2')
    dic3 = unpickle('../cifar-10-batches-py/data_batch_3')
    dic4 = unpickle('../cifar-10-batches-py/data_batch_4')
    dic5 = unpickle('../cifar-10-batches-py/data_batch_5')

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
        cv2.imwrite(f'../images/ori1-{i}.png', dic1b[i])

    for i in range(dic2b.shape[0]):
        cv2.imwrite(f'../images/ori2-{i}.png', dic2b[i])

    for i in range(dic3b.shape[0]):
        cv2.imwrite(f'../images/ori3-{i}.png', dic3b[i])

    for i in range(dic4b.shape[0]):
        cv2.imwrite(f'../images/ori4-{i}.png', dic4b[i])

    for i in range(dic5b.shape[0]):
        cv2.imwrite(f'../images/ori5-{i}.png', dic5b[i])

if __name__ == "__main__":
    main()