import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# load dataset

ds, ds_info = tfds.load('cats_vs_dogs', split=['train[:80%]', 'train[80%:]'], as_supervised=True, with_info=True,
                        shuffle_files=True, download=True)

print(ds)
print(ds_info)

train_ds = ds[0]
test_ds = ds[1]


def preprocess_image(image, label, size=(255, 255)):
    image = tf.image.resize(image, size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


custom_size = (255, 255)
train_ds_preprocess = train_ds.map(lambda image, label: preprocess_image(image, label, custom_size))
test_ds_preprocess = test_ds.map(lambda image, label: preprocess_image(image, label, custom_size))

# Now we need to extract out labels and data to pass to the training
batch_size = 32
train_ds_batched = train_ds_preprocess.batch(batch_size)
test_ds_batched = test_ds_preprocess.batch(batch_size)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(255, 255, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
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

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])

history = model.fit(
    train_ds_batched,
    epochs=10,
    validation_data=test_ds_batched,
)

model.save("model.keras")

# Plot training history
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()



