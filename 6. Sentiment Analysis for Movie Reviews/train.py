from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
import wandb

# Initialize wandb
wandb.init(project="Tensorflow certification practice", save_code=True, name="LSTM 2 Layers Bidirectional",
           group="NLP Sentiment Classification")

# Load the IMDB dataset
num_words = 10000
max_review_length = 1000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=128, input_length=max_review_length))
model.add(Bidirectional(LSTM(64, return_sequences=True)))  # Bidirectional LSTM layer with return_sequences=True
model.add(LSTM(64))  # Second LSTM layer
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 128
epochs = 10

# Use wandb_callback to log metrics
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          callbacks=[wandb.keras.WandbCallback(save_model=False)])

wandb.finish()
