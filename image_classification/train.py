import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np


batch_size = 128
epochs = 15
total_train=2000
total_val=2000
IMG_HEIGHT = 150
IMG_WIDTH = 150

# mkdir data && cd data && wget https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip
train_dir = "data/cats_and_dogs_filtered/train"
validation_dir = "data/cats_and_dogs_filtered/validation"

train_image_generator = ImageDataGenerator(
    rescale=1.0 / 255
)  # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1.0 / 255)  #

train_data_gen = train_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode="binary",
)

val_data_gen = validation_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode="binary",
)


model = Sequential(
    [
        Conv2D(
            16,
            3,
            padding="same",
            activation="relu",
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        ),
        MaxPooling2D(),
        Conv2D(32, 3, padding="same", activation="relu"),
        MaxPooling2D(),
        Conv2D(64, 3, padding="same", activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation="relu"),
        Dense(1),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size,
)
