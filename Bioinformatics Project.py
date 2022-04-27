# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 13:20:46 2021

@author: eagle
"""


import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
##import scikit as skimage
 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from skimage import exposure
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
 
import pathlib
from pathlib import Path
 
# Import Dataset
data_dir = pathlib.Path(r"C:\Users\eagle\anaconda\Lib\massjpg\massjpg1")
 
# Configure dataset
 
batch_size = 32
img_height = 180
img_width = 180
 
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  shuffle = "true",
  validation_split=0.8,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

 
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  shuffle = "true",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

 
# Configure the dataset for performance


AUTOTUNE = tf.data.AUTOTUNE
 
train_ds = train_ds.cache().shuffle(2).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
 
# Standardize the data
 
normalization_layer = layers.Rescaling(1./255)
 
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))
 
#Setup model
 
num_classes = 1000
 
  
model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
 ## layers.Dense(num_classes, activation= 'sigmoid')
  layers.Dense(num_classes, activation='softmax')
])
 
# Compile the model
 
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_ds, val_ds, validation_data=(train_ds,val_ds), epochs=150, batch_size=10)
 
model.summary()
 
# Feed model images
 
epochs=10


history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
 
# Graph training results
 
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
 
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs_range = range(epochs)
 
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
 
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
 
#Train Model
 
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
 
model.summary()
 
epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
 
# Graph Training Results
 
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
 
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs_range = range(epochs)
 
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
 
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
