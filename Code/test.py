import tensorflow as tf

dataset = tfds.load('open_images/v7', split='train')
for datum in dataset:
  image, bboxes = datum["image"], datum["bboxes"]