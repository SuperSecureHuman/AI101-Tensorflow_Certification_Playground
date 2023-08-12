import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model("../2. Basic CNN (Binary)/model.keras")

# Print the summary of the loaded model
print(model.summary())

# Path to sample image
path = "./img.png"

# Load image as a PIL image
image_pil = tf.keras.utils.load_img(path)

# Convert PIL image to a NumPy array
image_np = tf.keras.utils.img_to_array(image_pil)

# Preprocess the image
def preprocess_image(image, size=(255, 255)):
    # Resize the image to the specified size
    image = tf.image.resize(image, size)
    # Convert image to float and scale to a range between 0 and 1
    image = tf.cast(image, tf.float32) / 255.0
    return image

image_preprocessed = preprocess_image(image_np)

# Print the shape of the preprocessed image
print(image_preprocessed.shape)

# Batch the image
image_preprocessed_batched = tf.expand_dims(image_preprocessed, 0)

# Make predictions using the loaded model
preds = model.predict(image_preprocessed_batched)

# Apply sigmoid to get probability
probability = tf.sigmoid(preds)
print("Probability:", probability.numpy()[0][0])  # Print the probability value
