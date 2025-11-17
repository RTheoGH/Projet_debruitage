import cv2
import numpy as np
import pickle
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def main(val_count=1000, train_count=None):
    out_train_truth = os.path.join('..', 'allImages', 'train', 'truth')
    out_val_truth = os.path.join('..', 'allImages', 'validation', 'truth')
    os.makedirs(out_train_truth, exist_ok=True)
    os.makedirs(out_val_truth, exist_ok=True)

    batches = [
        unpickle('../data/cifar-10-batches-py/data_batch_1')[b'data'],
        unpickle('../data/cifar-10-batches-py/data_batch_2')[b'data'],
        unpickle('../data/cifar-10-batches-py/data_batch_3')[b'data'],
        unpickle('../data/cifar-10-batches-py/data_batch_4')[b'data'],
        unpickle('../data/cifar-10-batches-py/data_batch_5')[b'data'],
    ]

    arrays = [b.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1) for b in batches]
    data = np.concatenate(arrays, axis=0)

    total = data.shape[0]
    if val_count is None:
        val_count = 1000
    if val_count < 0:
        val_count = 0
    if train_count is None or train_count < 0:
        train_count = total - val_count

    if val_count > total:
        val_count = total
        train_count = 0
    if val_count + train_count > total:
        train_count = total - val_count

    val_idx = 1
    train_idx = 1
    for i in range(total):
        img = data[i]
        if i < val_count:
            fname = f'val{val_idx}.png'
            cv2.imwrite(os.path.join(out_val_truth, fname), img)
            val_idx += 1
        elif i < val_count + train_count:
            fname = f'train{train_idx}.png'
            cv2.imwrite(os.path.join(out_train_truth, fname), img)
            train_idx += 1
        else:
            break

if __name__ == "__main__":
    VAL_COUNT = 1000
    TRAIN_COUNT = 10000
    main(val_count=VAL_COUNT, train_count=TRAIN_COUNT)