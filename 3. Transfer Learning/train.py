import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow_hub as hub

## Load Dataset

# Load the 'cats_vs_dogs' dataset, splitting it into train and test sets
ds, ds_info = tfds.load('cats_vs_dogs', split=['train[:80%]', 'train[80%:]'], as_supervised=True, with_info=True,
                        shuffle_files=True, download=True)

# Separate the loaded dataset into train and test sets
train_ds = ds[0]
test_ds = ds[1]

# Print the class labels in the dataset
print(ds_info.features['label'].names)

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

# Print the shape of an image batch and its corresponding label batch
for image_batch, labels_batch in train_ds_batched:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

# Create a model using a pre-trained MobileNetV3 as feature extractor

model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_small_075_224/classification/5",
                   trainable=False, input_shape=(224, 224, 3))
])

# Pass an image batch through the model to get feature vectors
feature_batch = model(image_batch)
print(feature_batch.shape)

# Create the final model by adding a dense layer for classification

final_model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_small_075_224/classification/5", trainable=False, input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(1)
])

## Train the final model

# Define optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Compile the final model
final_model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])

# Train the final model
history = final_model.fit(
    train_ds_batched,
    epochs=1,
    validation_data=test_ds_batched,
)

# Save the trained final model
final_model.save("mobile_net_cats_dogs.keras")
