import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow_hub as hub

## Load Dataset
ds, ds_info = tfds.load('cats_vs_dogs', split=['train[:80%]', 'train[80%:]'], as_supervised=True, with_info=True,
                        shuffle_files=True, download=True)

train_ds = ds[0]
test_ds = ds[1]

print(ds_info.features['label'].names)


def preprocess_image(image, label, size=(255, 255)):
    image = tf.image.resize(image, size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


custom_size = (224, 224)
train_ds_preprocess = train_ds.map(lambda image, label: preprocess_image(image, label, custom_size))
test_ds_preprocess = test_ds.map(lambda image, label: preprocess_image(image, label, custom_size))

# Now we need to extract out labels and data to pass to the training
batch_size = 32
train_ds_batched = train_ds_preprocess.batch(batch_size)
test_ds_batched = test_ds_preprocess.batch(batch_size)

for image_batch, labels_batch in train_ds_batched:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_small_075_224/classification/5",
                   trainable=False,input_shape=(224, 224, 3))
])

feature_batch = model(image_batch)
print(feature_batch.shape)

# From here, we see that output dense layer has 1001 params

# create final model

final_model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_small_075_224/classification/5",trainable=False, input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(1)
])

## Train as usual

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

final_model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])

history = final_model.fit(
    train_ds_batched,
    epochs=1,
    validation_data=test_ds_batched,
)

final_model.save("mobile_net_cats_dogs.keras")

