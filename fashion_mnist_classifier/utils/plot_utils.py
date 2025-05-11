import matplotlib.pyplot as plt
import numpy as np

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot_sample_images(images, labels, predictions=None):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        label = class_names[labels[i]]
        if predictions is None:
            plt.xlabel(label)
        else:
            pred_label = class_names[np.argmax(predictions[i])]
            plt.xlabel(f"Pred: {pred_label}\nTrue: {label}")
    plt.tight_layout()
    plt.show()
