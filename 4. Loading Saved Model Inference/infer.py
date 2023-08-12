import tensorflow as tf

model = tf.keras.models.load_model("../2. Basic CNN (Binary)/model.keras")

print(model.summary())

# Path to sample image
path = "./img.png"

# Load image as a PIL image
image_pil = tf.keras.utils.load_img(path)

# Convert PIL image to a NumPy array
image_np = tf.keras.utils.img_to_array(image_pil)

# Preprocess the image
def preprocess_image(image, size=(255, 255)):
    image = tf.image.resize(image, size)
    image = tf.cast(image, tf.float32) / 255.0
    return image

image_preprocessed = preprocess_image(image_np)

print(image_preprocessed.shape)

# Batch the image
image_preprocessed_batched = tf.expand_dims(image_preprocessed, 0)

# Make predictions
preds = model.predict(image_preprocessed_batched)

# Apply sigmoid to get probability
probability = tf.sigmoid(preds)
print("Probability:", probability.numpy()[0][0])  # Print the probability value
