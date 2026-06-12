import struct
import idx2numpy
import numpy as np
import os

def save_fvecs(filename, arr):
    """
    Saves a NumPy array (n x d) to a .fvecs file.
    The .fvecs format is: for each vector, a 4-byte integer dimension,
    followed by the vector's floats.
    """
    print(f"Saving {arr.shape[0]} vectors to {filename}...")
    with open(filename, 'wb') as f:
        for row in arr:
            d = np.int32(row.shape[0])
            f.write(d.tobytes())
            f.write(row.astype(np.float32).tobytes())
    print("Done.")

# --- IMPORTANT: Set the path to your Kaggle MNIST folder ---
# Example: kaggle_mnist_path = "/home/username/downloads/mnist"
kaggle_mnist_path = "/home/nsyatov/Downloads/MNIST"
# ----------------------------------------------------------

# Define full paths to the .ubyte files
train_images_path = os.path.join(kaggle_mnist_path, "train-images.idx3-ubyte")
train_labels_path = os.path.join(kaggle_mnist_path, "train-labels.idx1-ubyte")
test_images_path = os.path.join(kaggle_mnist_path, "t10k-images.idx3-ubyte")
test_labels_path = os.path.join(kaggle_mnist_path, "t10k-labels.idx1-ubyte")

# Load data using idx2numpy
print("Loading training images...")
x_train = idx2numpy.convert_from_file(train_images_path)
print("Loading training labels...")
y_train = idx2numpy.convert_from_file(train_labels_path)
print("Loading test images...")
x_test = idx2numpy.convert_from_file(test_images_path)
print("Loading test labels...")
y_test = idx2numpy.convert_from_file(test_labels_path)

# Flatten the 28x28 images to 784-dim vectors and normalize pixel values to [0, 1]
print("Flattening and normalizing training data...")
x_train_flat = x_train.reshape(x_train.shape[0], -1).astype(np.float32) / 255.0
print("Flattening and normalizing test data...")
x_test_flat = x_test.reshape(x_test.shape[0], -1).astype(np.float32) / 255.0

# Save the data in .fvecs format
save_fvecs("mnist_base.fvecs", x_train_flat)   # 60,000 training vectors
save_fvecs("mnist_query.fvecs", x_test_flat)   # 10,000 test vectors