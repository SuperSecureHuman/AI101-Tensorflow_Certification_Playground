import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd
import os
import wandb

# Initialize wandb
wandb.init(project="Tensorflow certification practice", save_code=True , name="Temperature Prediction LSTM",
           group="Time Series")


zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)


# Load the CSV file into a DataFrame
data = pd.read_csv(csv_path)

# Extract the 'T (degC)' column, which corresponds to temperature
temperature_data = data['T (degC)']


# Define window size for sequence data
window_size = 30

# Create sequences and labels
sequences = []
labels = []

for i in range(len(temperature_data) - window_size):
    sequences.append(temperature_data[i:i+window_size])
    labels.append(temperature_data[i+window_size])

# Convert to numpy arrays
sequences = np.array(sequences)
labels = np.array(labels)

# Split the data into training and validation sets
split = int(0.8 * len(sequences))
train_sequences = sequences[:split]
train_labels = labels[:split]
val_sequences = sequences[split:]
val_labels = labels[split:]

# Build the LSTM model
model = Sequential([
    LSTM(64, input_shape=(window_size, 1)),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define a learning rate scheduler
def lr_schedule(epoch, lr):
    if epoch < 1:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

# Train the model
model.fit(train_sequences, train_labels, epochs=20, batch_size=32,
          validation_data=(val_sequences, val_labels), callbacks=[lr_scheduler, wandb.keras.WandbCallback(save_model=False)])

test_loss = model.evaluate(val_sequences, val_labels)

print(f"Test Loss: {test_loss}")

wandb.finish()