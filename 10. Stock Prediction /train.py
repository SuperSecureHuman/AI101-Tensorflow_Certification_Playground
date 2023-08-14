import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Bidirectional, GRU, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import os
import wandb

# Initialize wandb
wandb.init(project="Tensorflow certification practice", save_code=True , name="Stock Prediction",
           group="Time Series")

csv_file = tf.keras.utils.get_file(
    origin='https://github.com/rashida048/Datasets/raw/master/stock_data.csv',
    fname='stock_data.csv')
csv_path, _ = os.path.splitext(csv_file)

# Load the CSV file into a DataFrame
data = pd.read_csv("/home/venom/.keras/datasets/stock_data.csv")

features = data[['Open', 'High', 'Low']]
labels = data['Close']

# Normalize the data
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(features)
normalized_labels = scaler.fit_transform(labels.values.reshape(-1, 1))

# Define window size for sequence data
window_size = 30

# Create sequences and labels
sequences = []
labels_list = []

for i in range(len(normalized_features) - window_size):
    sequences.append(normalized_features[i:i+window_size])
    labels_list.append(normalized_labels[i+window_size])

# Convert to numpy arrays
sequences = np.array(sequences)
labels = np.array(labels_list)

# Split the data into training and validation sets
split = int(0.8 * len(sequences))
train_sequences = sequences[:split]
train_labels = labels[:split]
val_sequences = sequences[split:]
val_labels = labels[split:]

print(normalized_features.shape)

# Build the model
model = Sequential([
    SimpleRNN(64, return_sequences=True, input_shape=(window_size, normalized_features.shape[1])),
    LSTM(64, return_sequences=True),
    GRU(32),
    Dense(1)
])


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(train_sequences, train_labels, epochs=20, batch_size=32,
          validation_data=(val_sequences, val_labels),
          callbacks=[wandb.keras.WandbCallback(save_model=False)])

# Make predictions
predictions = model.predict(val_sequences)

# Denormalize predictions and labels
predictions = scaler.inverse_transform(predictions)
true_labels = scaler.inverse_transform(val_labels)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(true_labels, predictions)
print(f"Mean Absolute Error: {mae}")