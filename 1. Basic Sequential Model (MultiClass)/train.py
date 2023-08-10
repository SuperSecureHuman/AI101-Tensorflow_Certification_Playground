

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(tf.__version__)

train_df = pd.read_csv("../datasets/mnist_csv/mnist_train.csv")
test_df = pd.read_csv("../datasets/mnist_csv/mnist_test.csv")

# Prepare data

train_data = train_df.drop(labels=["label"], axis=1)
train_data = train_data.values
train_label = train_df["label"]

test_data = test_df.drop(labels=["label"], axis=1)
test_data = test_data.values
test_label = test_df["label"]

# Normalize data
train_data = train_data / 255.0
test_data = test_data / 255.0

## One Hot Encode Labels
one_hot_train_label = pd.get_dummies(train_label)
one_hot_train_label = one_hot_train_label.values
one_hot_test_label = pd.get_dummies(test_label)
one_hot_test_label = one_hot_test_label.values
print(test_data.shape, one_hot_test_label.shape)

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Define the sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10)  # No activation here
])

# Compile model - Adam optimizer
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# Train model
history = model.fit(train_data, one_hot_train_label,
                    validation_data=(test_data, one_hot_test_label),
                    epochs=10)

# Plot training history
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

