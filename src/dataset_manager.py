import os
import numpy as np
import matplotlib.pyplot as plt

class DatasetManager:
    def __init__(self, root_path):
        self.root_path = root_path

    def analyze_dataset(self, dataset_name, dataset_type):
        db = np.load(os.path.join(self.root_path, dataset_name, f'{dataset_type}.npz'))
        print(f"Dataset: {dataset_name} ({dataset_type})")
        print(f"Keys: {db.files}")

        embeddings_shape = db['embeddings'].shape
        labels_shape = db['labels'].shape
        print(f"Embeddings shape: {embeddings_shape}")
        print(f"Labels shape: {labels_shape}")

        unique_labels, counts = np.unique(db['labels'], return_counts=True)
        print(f"Number of unique labels: {len(unique_labels)}")
        print(f"Label distribution: {dict(zip(unique_labels, counts))}")

        # Plotting the label distribution
        plt.figure(figsize=(10, 6))
        plt.bar(unique_labels, counts)
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.title(f'Label Distribution in {dataset_name} ({dataset_type})')
        plt.xticks(unique_labels)
        plt.show()

        embeddings_mean = np.mean(db['embeddings'], axis=0)
        embeddings_std = np.std(db['embeddings'], axis=0)
        print(f"Embeddings mean: {embeddings_mean[:5]}...")
        print(f"Embeddings std: {embeddings_std[:5]}...\n")

    def load_dataset(self, dataset_name, dataset_type):
        return np.load(os.path.join(self.root_path, dataset_name, f'{dataset_type}.npz'))
