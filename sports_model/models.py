import tensorflow as tf
from tensorflow import keras
from keras import layers
def myModel():
    num_features = 47
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(8, 5, strides=1, activation='relu', padding='same', input_shape=(num_features, 1)))
    model.add(layers.Conv1D(16, 3, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv1D(32, 3, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv1D(128, 3, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

#model.summary()

def save_model(model, name):
    model.save(name + ".h5")

def load_model(name):
    model = keras.models.load_model(name + ".h5")
    return model
