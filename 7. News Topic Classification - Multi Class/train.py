import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense , Bidirectional
import wandb

# Initialize wandb
wandb.init(project="Tensorflow certification practice", save_code=True, name="LSTM_GRU",
           group="NLP News MultiClass Classification")

# Load the dataset
# link - https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Combine title and description for training and testing
train_texts = train_data["Title"] + " " + train_data["Description"]
test_texts = test_data["Title"] + " " + test_data["Description"]

# Convert class indices to one-hot encoding
num_classes = 4
train_labels = tf.keras.utils.to_categorical(train_data["Class Index"] - 1, num_classes=num_classes)

# Tokenize the text
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# Pad sequences to the same length
max_sequence_length = 100
train_sequences = pad_sequences(train_sequences, maxlen=max_sequence_length)
test_sequences = pad_sequences(test_sequences, maxlen=max_sequence_length)

# Build the model
embedding_dim = 100
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(Bidirectional(LSTM(64, return_sequences=True)))  # Bidirectional LSTM layer with return_sequences=True
model.add(Bidirectional(GRU(64)))  # Add a Bidirectional GRU layer
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with wandb callback
batch_size = 64
epochs = 10
model.fit(train_sequences, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=[wandb.keras.WandbCallback(save_model=False)])

# Evaluate the model
test_labels = tf.keras.utils.to_categorical(test_data["Class Index"] - 1, num_classes=num_classes)
loss, accuracy = model.evaluate(test_sequences, test_labels)
print(f'Test accuracy: {accuracy:.4f}')
wandb.log({"Test Accuracy": accuracy})

wandb.finish()