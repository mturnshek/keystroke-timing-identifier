import os

import tensorflow as tf
import numpy as np

from utils import get_class_weights


class Identifier(tf.keras.Model):
    def __init__(self):
        self.num_classes = 16
        super(Identifier, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=5,
            strides=1,
            padding='valid',
            input_shape=(None, 256, 1)
        )
        # print(self.conv1.input_shape)
        self.flat1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.final = tf.keras.layers.Dense(self.num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.flat1(x)
        x = self.dense1(x)
        x = self.final(x)
        return x

    def train(self, epochs=500):
        opt = tf.keras.optimizers.Adam()
        loss = tf.keras.losses.categorical_crossentropy
        X, Y = self.get_dataset()

        self.compile(optimizer=opt, loss=loss)
        self.fit(
            X,
            Y,
            batch_size=64,
            epochs=epochs,
            class_weight=get_class_weights(np.argmax(Y, axis=1), smooth_factor=0.05) # weight under-represented classes almost equally
        )

    def get_dataset(self):
        X = []
        Y = []
        name_no = 0
        for name in os.listdir('data'):
            for filename in os.listdir(f'data/{name}/X'):
                x = np.load(f'data/{name}/X/{filename}')
                y = np.zeros(self.num_classes)
                y[name_no] = 1
                X.append(x)
                Y.append(y)
            name_no += 1
            assert(name_no < self.num_classes), "The number of classes in the dataset exceeds the number of classes in the model."
        X = np.array(X)
        Y = np.array(Y)
        return (X, Y)