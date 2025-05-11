from fpdf import FPDF
import os

# Create PDF class
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(200, 10, txt="Fashion MNIST Classifier Report", ln=True, align='C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(200, 10, txt=title, ln=True)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_code(self, code):
        self.set_font('Courier', '', 10)
        self.multi_cell(0, 10, code)
        self.ln()

    def add_image(self, image_path, caption=""):
        self.ln(10)
        if os.path.exists(image_path):
            self.image(image_path, x=10, y=None, w=180)
            self.ln(5)
            self.cell(200, 10, txt=caption, ln=True)

# Initialize PDF
pdf = PDF()
pdf.add_page()

# Title
pdf.chapter_title("Project Overview")
pdf.chapter_body("This project is about classifying images from the Fashion MNIST dataset using a deep learning model built with TensorFlow and Keras.")

# Add Preprocessing Data Section
pdf.chapter_title("1. Preprocessing Data")

# Preprocessing code
preprocessing_code = '''# Load the Fashion MNIST dataset
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the images to a range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Print the shape of the data
print(f"Train images shape: {train_images.shape}")
print(f"Test images shape: {test_images.shape}")
'''
pdf.add_code(preprocessing_code)

# Add Model Code Section
pdf.chapter_title("2. Model Code")

# Model code
model_code = '''# Build the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc}")
'''
pdf.add_code(model_code)

# Add Generated Output Section
pdf.chapter_title("3. Generated Output")

# Add output content
output_content = '''The model was trained for 5 epochs on the Fashion MNIST dataset. The accuracy and loss on the test data are as follows:
- Test Accuracy: 0.8807
- Test Loss: 0.3335
'''

pdf.chapter_body(output_content)

# Add Training Plot (Example)
pdf.chapter_title("4. Training Loss Plot")

# Replace this with the actual path to your plot
pdf.add_image("path_to_your_training_loss_plot.png", "Training Loss vs. Epochs")

# Output PDF
pdf.output("fashion_mnist_report.pdf")
