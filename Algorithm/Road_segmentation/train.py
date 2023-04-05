import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import Model
from keras.applications import VGG16
from matplotlib import pyplot as plt
import keras.callbacks
import cv2

# SET GPU
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

SEED = 123
IMG_SIZE = 224
BUFFER_SIZE = 1000
BATCH_SIZE = 2
AUTOTUNE = tf.data.AUTOTUNE

file_data = "data/data.csv"
data = pd.read_csv(file_data)
data = data.to_numpy()

TRAIN_SIZE = 10
VAL_SIZE = 10
TEST_SIZE = 10

@tf.function
def read_img(data):
    # Input image
    img = tf.io.read_file(data[0])
    img = tf.image.decode_jpeg(img)
    # Output segment
    segment = tf.io.read_file(data[1])
    segment = tf.image.decode_jpeg(segment)
    rgb_road = tf.constant([255, 0, 255], dtype=tf.uint8)
    segment = tf.reduce_all(segment == rgb_road, axis= 2)
    segment = tf.cast(segment, dtype=tf.uint8)
    segment_shape = tf.shape(segment)
    segment = tf.reshape(segment, (segment_shape[0], segment_shape[1], 1))
    return img, segment

dataset = tf.data.Dataset.from_tensor_slices(data)
dataset = dataset.map(read_img)

dataset = dataset.shuffle(BUFFER_SIZE, seed= SEED)
train_data = dataset.take(TRAIN_SIZE)
val_data = dataset.skip(TRAIN_SIZE).take(VAL_SIZE)
test_data = dataset.skip(TRAIN_SIZE + VAL_SIZE).take(TEST_SIZE)

# Create a generator.
rng = tf.random.Generator.from_seed(SEED, alg='philox')

@tf.function
def resize_and_rescale(img:tf.Tensor, segment:tf.Tensor):
    img = tf.cast(img, tf.float32)
    img = img / 255.0
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    segment = tf.image.resize(segment, (IMG_SIZE, IMG_SIZE))
    #img = tf.expand_dims(img, axis=0)
    #segment = tf.expand_dims(segment, axis= 0)
    return img , segment

@tf.function
def preprocess(ds:tf.data.Dataset, augment=False):
    ds = ds.map(resize_and_rescale, num_parallel_calls= AUTOTUNE)
    #ds = ds.batch(BATCH_SIZE)
    return ds

# Set up data
# train_data = train_data.repeat(3)
train_data = preprocess(train_data)
val_data = preprocess(val_data)
test_data = preprocess(test_data)
train_data = train_data.batch(BATCH_SIZE)

inputs = layers.Input((224, 224, 3))

outputs = layers.Conv2D(1, 1, activation= 'sigmoid')(inputs)

model = Model(inputs, outputs)

model.compile(
    optimizer= keras.optimizers.Adam(),
    loss= keras.losses.BinaryCrossentropy(),
    metrics= keras.metrics.MeanIoU(2),
)
train_data
model.fit(
    train_data,
    epochs= 2,
)