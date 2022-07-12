# Cuong dep trai sieu cap vjp pr0
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from keras import models

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

model = keras.Sequential()
model.add(layers.Input(28*28))
model.add(layers.Dense(126, activation="relu"))
model.add(layers.Dense(126, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

model.summary()

model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits= False), 
                optimizer = keras.optimizers.Adam(learning_rate=0.001), 
                metrics = ["accuracy"])

model.fit(x_train, y_train, epochs = 5)

