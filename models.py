import tensorflow as tf
from tensorflow import keras
from keras import layers
def myModel():
    num_features = 47
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(8, 3, strides=1, activation='relu', input_shape=(num_features, 1), use_bias=False))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv1D(16, 3, strides=1, activation='relu', use_bias=False))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv1D(32, 3, strides=1, activation='relu', use_bias=False))
    model.add(layers.BatchNormalization())


    model.add(layers.Conv1D(1, 3, strides=1, activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    model.add(layers.Dropout(0.1))
    model.add(layers.Activation('sigmoid'))
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics='acc')
    return model

#model.summary()

def save_model(model, name):
    model.save(name + ".h5")

def load_model(name):
    model = keras.models.load_model(name + ".h5")
    return model
