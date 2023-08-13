import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import wandb
from wandb.keras import WandbCallback

# Initialize WandB run
wandb.init(project="Tensorflow certification practice", group="CNN", save_code=True, name="Transfer Learning Cats vs Dogs")

## Load Dataset

# Load the 'cats_vs_dogs' dataset, splitting it into train and test sets
ds, ds_info = tfds.load('cats_vs_dogs',
                        split=['train[:80%]', 'train[80%:]'],
                        as_supervised=True, with_info=True,
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
custom_size = (224, 224)

# Apply image preprocessing to train and test datasets
train_ds_preprocess = train_ds.map(lambda image, label: preprocess_image(image, label, custom_size))
test_ds_preprocess = test_ds.map(lambda image, label: preprocess_image(image, label, custom_size))

# Batch the preprocessed datasets for training and testing
batch_size = 32
train_ds_batched = train_ds_preprocess.batch(batch_size)
test_ds_batched = test_ds_preprocess.batch(batch_size)

# Create a model using a pre-trained MobileNetV3 as feature extractor

model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_small_075_224/classification/5",
                   trainable=False, input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(1)
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
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

# Close the WandB run
wandb.finish()
