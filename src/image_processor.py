import joblib
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import random
import os

class ImageProcessor:
    def __init__(self, data_full_path):
        self.cifar10_images = []
        self.data_full_path = data_full_path

    def process_images_in_chunks(self, img_filenames, chunk_size):
        print("Processing images in chunks...")
        for i in range(0, len(img_filenames), chunk_size):
            chunk_filenames = img_filenames[i:i + chunk_size]
            for img_filename in chunk_filenames:
                print(f"Loading images from {img_filename}...")
                with open(img_filename, 'rb') as f:
                    self.cifar10_images.extend(joblib.load(f))
        print(f"Total images loaded: {len(self.cifar10_images)}")

    def find_interesting_indices(self, x_embedded, y_train, threshold=10):
        distances_file = os.path.join(self.data_full_path, 'distances.pkl')
        interesting_indices_file = os.path.join(self.data_full_path, 'interesting_indices.pkl')

        if os.path.exists(distances_file) and os.path.exists(interesting_indices_file):
            print("Loading precomputed distances and interesting indices...")
            distances = joblib.load(distances_file)
            interesting_indices = joblib.load(interesting_indices_file)
        else:
            print("Calculating distances...")
            distances = euclidean_distances(x_embedded, x_embedded)
            interesting_indices = []
            print("Find interesting indicies...")
            for idx in range(len(x_embedded)):
                neighbor_labels = y_train[np.argsort(distances[idx])[1:threshold+1]]
                if len(set(neighbor_labels)) > 6:  # Increase the threshold for more selectivity
                    interesting_indices.append(idx)
            
            # Save the computed distances and interesting indices for future use
            print(f"Saving distances to {distances_file} and interesting indices to {interesting_indices_file}...")
            joblib.dump(distances, distances_file)
            joblib.dump(interesting_indices, interesting_indices_file)

        print(f"Total interesting indices found: {len(interesting_indices)}")
        return interesting_indices

    def analyze_interesting_point(self, x_embedded, y_train, cifar10_images, interesting_indices, index=-1):
        if index == -1:
            idx = random.choice(interesting_indices)
            print(f"Randomly selected interesting index: {idx}")
        elif 0 <= index < len(x_embedded):
            idx = index
            print(f"Selected specific index: {idx}")
        else:
            raise ValueError(f"Index out of bounds. Choose a valid index between 0 and {len(x_embedded) - 1}")

        distances_file = os.path.join(self.data_full_path, 'distances.pkl')
        distances = joblib.load(distances_file)
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

        print(f"Index {idx} has nearest neighbors: {nearest_indices}")
        return idx, nearest_indices
