import joblib
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import random

class ImageProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.cifar10_images = []

    def process_images_in_chunks(self, img_filenames, chunk_size):
        for i in range(0, len(img_filenames), chunk_size):
            chunk_filenames = img_filenames[i:i + chunk_size]
            for img_filename in chunk_filenames:
                print(f"Loading images from {img_filename}...")
                with open(img_filename, 'rb') as f:
                    loaded_images = joblib.load(f)
                    print(f"Loaded images shape: {np.array(loaded_images).shape}")
                    self.cifar10_images.extend(loaded_images)

    def find_interesting_indices(self, x_embedded, y_train, threshold=5):
        print("Finding interesting indices...")
        distances = euclidean_distances(x_embedded, x_embedded)
        interesting_indices = []
        for idx in range(len(x_embedded)):
            neighbor_labels = y_train[np.argsort(distances[idx])[1:threshold+1]]
            if len(set(neighbor_labels)) > 2:
                interesting_indices.append(idx)
        print(f"Total interesting indices found: {len(interesting_indices)}")
        return interesting_indices

    def analyze_interesting_point(self, x_embedded, y_train, cifar10_images, interesting_indices, index=-1):
        if index == -1:
            if not interesting_indices:
                raise ValueError("No interesting indices found.")
            idx = random.choice(interesting_indices)
        elif 0 <= index < len(x_embedded):
            idx = index
        else:
            raise ValueError(f"Index out of bounds. Choose a valid index between 0 and {len(x_embedded) - 1}")

        distances = euclidean_distances(x_embedded, x_embedded)
        sorted_indices = np.argsort(distances[idx])

        nearest_indices = []
        for i in sorted_indices[1:]:
            if i < len(cifar10_images):
                nearest_indices.append(i)
            if len(nearest_indices) >= 5:
                break

        if len(nearest_indices) < 5:
            raise ValueError("Not enough valid nearest neighbors found.")

        if idx >= len(cifar10_images):
            raise ValueError(f"Selected index {idx} is out of bounds for cifar10_images with length {len(cifar10_images)}.")

        return idx, nearest_indices
