import numpy as np
import tensorflow as tf
from models import myModel, save_model, load_model


data = np.load('normData.npy')


num_data = data.shape[0]
X, y = data[:, :-1], data[:, -1]
X = X.reshape((num_data, X.shape[1], 1))
idxs = np.random.permutation(num_data)
X, y = X[idxs], y[idxs]
print(X.shape, y.shape)




model = myModel()
model.summary()


history = model.fit(x=X, y=y, epochs=1000, validation_split=0.3, batch_size=128, shuffle=True, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=200, restore_best_weights=True)])
save_model(model, "model")


# Plot the model history with matplotlib.pyplot.
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()



plt.plot(history.history['acc'], label='val_acc')
plt.plot(history.history['val_acc'], label='acc')
plt.legend()
plt.show()





