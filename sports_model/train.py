import numpy as np
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5200)])
        print("GPU found and set to 5200MB")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found")

from .models import myModel, save_model, load_model
from .utils import apply_random_noise, apply_feature_permutation, apply_feature_swapping, load_data
import matplotlib.pyplot as plt


def apply_augmentation(X, y):
    # Apply data augmentation
    augmented_X = []
    augmented_y = []
    idxs = np.random.permutation(X.shape[0])
    X = X[idxs]
    y = y[idxs]
    augmented_X.extend(X)
    for _ in range(3):
        augmented_X.extend(apply_feature_swapping(X, y))
        augmented_X.extend(apply_random_noise(X))
        augmented_X.extend(apply_feature_permutation(X))
    for _ in range(9+1):
        augmented_y.extend(y)

    augmented_idxs = np.random.permutation(len(augmented_X))
    augmented_X = np.array(augmented_X)
    augmented_X = augmented_X[augmented_idxs]
    augmented_y = np.array(augmented_y)
    augmented_y = augmented_y[augmented_idxs]

    return augmented_X, augmented_y


def train():
    X_train, y_train, X_test, y_test = load_data()

    model = myModel()
    #model.summary()

    optimizer = tf.keras.optimizers.Adamax(learning_rate=1e-2, epsilon=1e-3)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    train_acc_metric = tf.keras.metrics.BinaryAccuracy()
    val_acc_metric = tf.keras.metrics.BinaryAccuracy()

    losses = []
    accuracies = []
    val_accuracies = []
    best_val_acc = 0
    bad_count=0
    for epoch in range(20000):
        X_aug, y_aug = apply_augmentation(X_train, y_train)
        epoch_losses = []
        epoch_accuracies_train = []

        for i in range(0, len(X_aug), 128):
            X_batch = X_aug[i:i+128]
            y_batch = y_aug[i:i+128]
            with tf.GradientTape() as tape:
                logits = model(X_batch, training=True)
                loss_value = loss_fn(y_batch, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            train_acc_metric.update_state(tf.reshape(y_batch, logits.shape), logits) # type: ignore

            epoch_losses.append(loss_value.numpy())
            epoch_accuracies_train.append(train_acc_metric.result().numpy())
        losses.append(np.mean(epoch_losses))
        accuracies.append(np.mean(epoch_accuracies_train))

        #train_acc = train_acc_metric.result()
        train_acc_metric.reset_states()

        epoch_accuracies_val = []
        val_logits = model(X_test, training=False)
        val_acc_metric.update_state(tf.reshape(y_test, val_logits.shape), val_logits) # type: ignore

        epoch_accuracies_val.append(val_acc_metric.result().numpy())
        val_acc_metric.reset_states()

        val_accuracies.append(epoch_accuracies_val[-1])
        if epoch_accuracies_val[-1] > best_val_acc:
            best_val_acc = epoch_accuracies_val[-1]
            save_model(model, "best_model")
            print('Best model saved', best_val_acc)
            bad_count = 0
        else:
            bad_count += 1
            if bad_count > 50:
                break
        

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: train_acc={accuracies[-1] :.4f}, val_acc={val_accuracies[-1]:.4f}")

    return losses, accuracies, val_accuracies


def visualize_training(losses, accuracies, val_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(losses, label='loss')
    ax2.plot(accuracies, label='accuracy')
    ax2.plot(val_accuracies, label='val_accuracy')
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    losses, accuracies, val_accuracies = train()
    visualize_training(losses, accuracies, val_accuracies)