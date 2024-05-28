import joblib
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import random

class ImageProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.images = []

    def process_images_in_chunks(self, img_filenames, chunk_size):
        """Load images in chunks to manage memory usage."""
        for i in range(0, len(img_filenames), chunk_size):
            chunk_filenames = img_filenames[i:i + chunk_size]
            for img_filename in chunk_filenames:
                loaded_images = joblib.load(img_filename)
                self.images.extend(loaded_images)

    def find_interesting_indices(self, x_embedded, y_train):
        """Find indices of images with diverse neighbors."""
        distances = euclidean_distances(x_embedded, x_embedded)
        interesting_indices = []

        for threshold in range(5, 0, -1):
            for idx in range(len(x_embedded)):
                neighbor_labels = y_train[np.argsort(distances[idx])[1:]]
                unique_labels = set(neighbor_labels[:threshold])
                if len(unique_labels) >= threshold:
                    interesting_indices.append(idx)
            if len(interesting_indices) >= 10:
                break

        return interesting_indices

    def analyze_interesting_point(self, x_embedded, y_train, images, interesting_indices, index=-1):
        """Analyze a specific point and its nearest neighbors."""
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
        for threshold in range(5, 0, -1):
            nearest_indices = [i for i in sorted_indices[1:] if i < len(images) and len(set(y_train[np.argsort(distances[i])[1:threshold+1]])) >= threshold]
            if len(nearest_indices) >= 10:
                print(f"Number of nearest points found: {len(nearest_indices)}")
                break

        if len(nearest_indices) < 5:
            raise ValueError("Not enough valid nearest neighbors found.")

        if idx >= len(images):
            raise ValueError(f"Selected index {idx} is out of bounds for images with length {len(images)}.")

        return idx, nearest_indices
