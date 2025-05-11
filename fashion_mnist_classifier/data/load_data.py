import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

# Load the dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the images (pixel values scaled to 0–1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Print shapes to verify
print(f"Train images shape: {train_images.shape}")
print(f"Test images shape: {test_images.shape}")
