from data.load_data import load_fashion_mnist
from models.fashion_model import build_model
from utils.plot_utils import plot_sample_images

# Load data
(x_train, y_train), (x_test, y_test) = load_fashion_mnist()

# Visualize data
plot_sample_images(x_train, y_train)

# Build and train model
model = build_model()
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Predict and visualize
predictions = model.predict(x_test[:25])
plot_sample_images(x_test[:25], y_test[:25], predictions)
