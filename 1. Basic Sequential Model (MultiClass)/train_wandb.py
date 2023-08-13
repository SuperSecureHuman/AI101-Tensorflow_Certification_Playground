import tensorflow as tf
import pandas as pd
import wandb
from wandb.keras import WandbCallback

# Initialize WandB run
wandb.init(project="Tensorflow certification practice", group="Basic", save_code=True, name="MNIST Sequential MultiClass")

# Check TensorFlow version
print(tf.__version__)

# Load the training and test datasets
train_df = pd.read_csv("../datasets/mnist_csv/mnist_train.csv")
test_df = pd.read_csv("../datasets/mnist_csv/mnist_test.csv")

# Prepare data

# Extract features from the training data
train_data = train_df.drop(labels=["label"], axis=1)
train_data = train_data.values

# Extract labels from the training data
train_label = train_df["label"]

# Extract features from the test data
test_data = test_df.drop(labels=["label"], axis=1)
test_data = test_data.values

# Extract labels from the test data
test_label = test_df["label"]

# Normalize data to a range between 0 and 1
train_data = train_data / 255.0
test_data = test_data / 255.0

# One Hot Encode Labels

# Convert training labels to one-hot encoded format
one_hot_train_label = pd.get_dummies(train_label)
one_hot_train_label = one_hot_train_label.values

# Convert test labels to one-hot encoded format
one_hot_test_label = pd.get_dummies(test_label)
one_hot_test_label = one_hot_test_label.values

# Print the shapes of test data and one-hot encoded test labels
print(test_data.shape, one_hot_test_label.shape)

# Define class names for visualization
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Define the sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10)  # No activation here
])

# Compile the model using the Adam optimizer, categorical cross-entropy loss, and accuracy metric
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Create a WandB callback
wandb_callback = WandbCallback(save_model=False)

# Train the model with the WandB callback
history = model.fit(
    train_data, one_hot_train_label,
    validation_data=(test_data, one_hot_test_label),
    epochs=10,
    callbacks=[wandb_callback]
)

# Log evaluation metrics
test_loss, test_accuracy = model.evaluate(test_data, one_hot_test_label)
wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})

# Log example predictions
num_examples = 10
test_predictions = model.predict(test_data[:num_examples])
test_pred_labels = [class_names[pred.argmax()] for pred in test_predictions]
test_true_labels = [class_names[label.argmax()] for label in one_hot_test_label[:num_examples]]

images = []
for i in range(num_examples):
    image = wandb.Image(test_data[i].reshape(28, 28))
    images.append(image)

wandb.log({"examples": [wandb.Image(image) for image in images],
           "predicted_labels": test_pred_labels,
           "true_labels": test_true_labels})


# Close the WandB run
wandb.finish()
