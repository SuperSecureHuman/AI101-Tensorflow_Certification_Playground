import tensorflow as tf
import tensorflow_datasets as tfds
import wandb
from wandb.keras import WandbCallback

# Initialize WandB run
wandb.init(project="Tensorflow certification practice",  save_code=True, group="CNN",
           name="CNN Binary Cats vs Dogs")

# Load dataset using TensorFlow Datasets

# Load the 'cats_vs_dogs' dataset, splitting it into train and test sets
ds, ds_info = tfds.load('cats_vs_dogs', split=['train[:80%]', 'train[80%:]'], as_supervised=True, with_info=True,
                        shuffle_files=True, download=True)

# Separate the loaded dataset into train and test sets
train_ds = ds[0]
test_ds = ds[1]

# Function to preprocess images

def preprocess_image(image, label, size=(255, 255)):
    # Resize the image to the specified size
    image = tf.image.resize(image, size)
    # Convert image to float and scale to a range between 0 and 1
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Define custom size for preprocessed images
custom_size = (255, 255)

# Apply image preprocessing to train and test datasets
train_ds_preprocess = train_ds.map(lambda image, label: preprocess_image(image, label, custom_size))
test_ds_preprocess = test_ds.map(lambda image, label: preprocess_image(image, label, custom_size))

# Batch the preprocessed datasets for training and testing
batch_size = 32
train_ds_batched = train_ds_preprocess.batch(batch_size)
test_ds_batched = test_ds_preprocess.batch(batch_size)

# Define the CNN model

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(255, 255, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])

# Create a WandB callback
wandb_callback = WandbCallback(save_model=False)

# Train the model with the WandB callback
history = model.fit(
    train_ds_batched,
    epochs=10,
    validation_data=test_ds_batched,
    callbacks=[wandb_callback]
)

# Log evaluation metrics
test_loss, test_accuracy = model.evaluate(test_ds_batched)
wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})

# Log example predictions
num_examples = 5
test_samples = test_ds_preprocess.take(num_examples)
images = [image.numpy() for image, _ in test_samples]

predicted_labels = []
for image, _ in test_samples:
    prediction = model.predict(tf.expand_dims(image, axis=0))[0]
    predicted_labels.append("Dog" if prediction > 0 else "Cat")

wandb.log({"example_images": [wandb.Image(image) for image in images],
           "predicted_labels": predicted_labels})

# Close the WandB run
wandb.finish()
