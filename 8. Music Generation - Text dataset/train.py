import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D
import wandb

# Initialize wandb
wandb.init(project="Tensorflow certification practice", save_code=True , name="EdSheeran Lyrics Generation",
           group="Text generation")

# Load the dataset
dataset = pd.read_csv("EdSheeran.csv")

# Clean the "Lyric" column: convert NaN to empty string and non-string values to strings
dataset["Lyric"] = dataset["Lyric"].apply(lambda x: str(x) if isinstance(x, (str, float)) else "")

# Combine lyrics text
lyrics_text = "\n".join(dataset["Lyric"])

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([lyrics_text])
total_words = len(tokenizer.word_index) + 1

# Convert text to sequences
sequences = tokenizer.texts_to_sequences([lyrics_text])[0]

# Create input sequences and corresponding target words
input_sequences = []
target_words = []
sequence_length = 10

for i in range(sequence_length, len(sequences)):
    input_sequences.append(sequences[i - sequence_length:i])
    target_words.append(sequences[i])

input_sequences = np.array(input_sequences)
target_words = np.array(target_words)

# Build and compile the model
embedding_dim = 128
model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=embedding_dim, input_length=sequence_length))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(GRU(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(GlobalMaxPooling1D())
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Train the model
epochs = 50
model.fit(input_sequences, target_words, epochs=epochs, callbacks=[wandb.keras.WandbCallback(save_model=False)])


# Generate text using the trained model
def generate_text(seed_text, next_words, model, tokenizer, max_sequence_length):
    generated_text = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length, padding='pre')

        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_index = np.argmax(predicted_probs)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        generated_text += " " + output_word
    return generated_text


generated_lyrics = generate_text(seed_text="I'm feeling", next_words=50, model=model,
                                 tokenizer=tokenizer, max_sequence_length=sequence_length)

print(generated_lyrics)

# Log the generated lyrics to WandB
wandb.log({"Generated Lyrics": generated_lyrics})

wandb.finish()
