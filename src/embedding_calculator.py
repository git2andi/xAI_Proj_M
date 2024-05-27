import os
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE

class EmbeddingCalculator:
    def __init__(self, root_path, data_full_path):
        self.root_path = root_path
        self.data_full_path = data_full_path
        os.makedirs(self.data_full_path, exist_ok=True)

    def calculate_and_save_embeddings_full(self, dataset_name, train_c10):
        pkl_filename = os.path.join(self.data_full_path, f'x_embedded_{dataset_name}.pkl')
        img_list_filename = os.path.join(self.data_full_path, f'cifar10_image_batches_{dataset_name}.pkl')

        if os.path.exists(pkl_filename) and os.path.exists(img_list_filename):
            print(f"Embeddings and image batches already exist for {dataset_name}. Skipping calculation.")
            return
        
        print("Loading embeddings and labels from dataset...")
        db_train = np.load(os.path.join(self.root_path, dataset_name, 'train.npz'))
        X_train, y_train = db_train['embeddings'], db_train['labels'].reshape(-1,)

        # Check if the sizes match
        if len(X_train) != len(train_c10):
            raise ValueError(f"Mismatch between full embeddings ({len(X_train)}) and images ({len(train_c10)}).")
        print(f"Full embeddings and images have matching lengths: {len(X_train)}")

        # Convert to PyTorch tensor
        X_train = torch.tensor(X_train)
        y_train = torch.tensor(y_train)

        # Create a combined dataset and dataloader
        combined_dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(combined_dataset, batch_size=2048, shuffle=False, num_workers=2)

        x_embedded_list = []
        labels_list = []
        img_filenames = []

        for i, (embeddings, labels) in enumerate(dataloader):
            # Perform t-SNE on the current batch of embeddings
            print(f"Processing batch {i + 1}/{len(dataloader)}...")
            batch_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(embeddings.numpy())
            x_embedded_list.append(batch_embedded)
            labels_list.append(labels.numpy())

            # Save the corresponding batch of images
            image_batch = []
            for j in range(len(embeddings)):
                index = i * dataloader.batch_size + j
                if index >= len(train_c10):
                    print(f"Index {index} out of range for train_c10 with length {len(train_c10)}")
                    break
                image, _ = train_c10[index]
                image_batch.append(image.numpy())
            
            batch_filename = os.path.join(self.data_full_path, f'cifar10_images_batch_{i}.pkl')
            print(f"Saving image batch {i} to {batch_filename}...")
            with open(batch_filename, 'wb') as f:
                joblib.dump(np.array(image_batch), f, compress=3, protocol=4)
            img_filenames.append(batch_filename)

        # Combine all batches
        x_embedded = np.vstack(x_embedded_list)
        labels_combined = np.concatenate(labels_list)

        # Save t-SNE embeddings and labels
        data_to_save = (x_embedded, labels_combined)
        print(f"Saving t-SNE embeddings and labels to {pkl_filename}...")
        with open(pkl_filename, 'wb') as f:
            joblib.dump(data_to_save, f, compress=3, protocol=4)

        # Save the list of image batch filenames
        img_list_filename = os.path.join(self.data_full_path, f'cifar10_image_batches_{dataset_name}.pkl')
        with open(img_list_filename, 'wb') as f:
            joblib.dump(img_filenames, f, compress=3, protocol=4)

        print(f"Files saved to {self.data_full_path}")

    def load_and_process_images(self, dataset_name):
        print("Loading embeddings and labels...")
        x_embedded, y_train = joblib.load(os.path.join(self.data_full_path, f'x_embedded_{dataset_name}.pkl'))
        img_list_filename = os.path.join(self.data_full_path, f'cifar10_image_batches_{dataset_name}.pkl')

        # Align x_embedded and y_train
        if x_embedded.shape[0] != y_train.shape[0]:
            print("Reshape was required")
            min_samples = min(x_embedded.shape[0], y_train.shape[0])
            x_embedded = x_embedded[:min_samples]
            y_train = y_train[:min_samples]

        print(f'x_embedded shape: {x_embedded.shape}')
        print(f'y_train shape: {y_train.shape}')

        # Load the list of image batch filenames
        img_filenames = joblib.load(img_list_filename)

        # Initialize a list to hold all images incrementally
        cifar10_images = []

        # Process images in smaller chunks to reduce RAM usage
        def process_images_in_chunks(img_filenames, chunk_size):
            for i in range(0, len(img_filenames), chunk_size):
                chunk_filenames = img_filenames[i:i + chunk_size]
                for img_filename in chunk_filenames:
                    print(f"Add {img_filename} to combine Batches...")
                    with open(img_filename, 'rb') as f:
                        loaded_images = joblib.load(f)
                        cifar10_images.extend(loaded_images)

        # Process the images in chunks
        process_images_in_chunks(img_filenames, chunk_size=1)

        return x_embedded, y_train, cifar10_images
