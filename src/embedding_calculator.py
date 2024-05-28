import os
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE

class EmbeddingCalculator:
    def __init__(self, root_path, data_path):
        self.root_path = root_path
        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)

    def calculate_and_save_embeddings(self, dataset_name, train_dataset, mode, random_state=42):
        """Calculate and save embeddings."""
        pkl_filename = os.path.join(self.data_path, f'x_embedded_{dataset_name}_{mode}.pkl')
        img_list_filename = os.path.join(self.data_path, f'{dataset_name}_image_batches_{mode}.pkl')

        if os.path.exists(pkl_filename) and os.path.exists(img_list_filename):
            print(f"Embeddings and image batches already exist for {dataset_name} ({mode}). Skipping calculation.")
            return
        
        print("Loading embeddings and labels from dataset...")
        db_train = np.load(os.path.join(self.root_path, dataset_name, 'train.npz'))
        X_train, y_train = db_train['embeddings'], db_train['labels'].reshape(-1,)

        if mode == "subsampled":
            X_train = X_train[::10]
            y_train = y_train[::10]
            train_dataset = [train_dataset[i] for i in range(len(train_dataset)) if i % 10 == 0]

        if len(X_train) != len(train_dataset):
            raise ValueError(f"Mismatch between embeddings ({len(X_train)}) and images ({len(train_dataset)}) for {mode} mode.")
        print(f"{mode.capitalize()} embeddings and images have matching lengths: {len(X_train)}")

        X_train = torch.tensor(X_train)
        y_train = torch.tensor(y_train)

        combined_dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(combined_dataset, batch_size=2048, shuffle=False, num_workers=2)

        x_embedded_list = []
        labels_list = []
        img_filenames = []

        for i, (embeddings, labels) in enumerate(dataloader):
            print(f"Processing batch {i + 1}/{len(dataloader)}...")
            tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=50, random_state=random_state)
            batch_embedded = tsne.fit_transform(embeddings.numpy())
            x_embedded_list.append(batch_embedded)
            labels_list.append(labels.numpy())

            image_batch = []
            for j in range(len(embeddings)):
                index = i * dataloader.batch_size + j
                if index >= len(train_dataset):
                    print(f"Index {index} out of range for train_dataset with length {len(train_dataset)}")
                    break
                image, _ = train_dataset[index]
                image_batch.append(image.numpy())
            
            batch_filename = os.path.join(self.data_path, f'{dataset_name}_images_batch_{mode}_{i}.pkl')
            joblib.dump(np.array(image_batch), batch_filename, compress=3)
            img_filenames.append(batch_filename)

        x_embedded = np.vstack(x_embedded_list)
        labels_combined = np.concatenate(labels_list)

        data_to_save = (x_embedded, labels_combined)
        joblib.dump(data_to_save, pkl_filename, compress=3)
        joblib.dump(img_filenames, img_list_filename, compress=3)

        print(f"Files saved to {self.data_path}")

    def load_and_process_images(self, dataset_name, mode):
        """Load and process images."""
        print("Loading embeddings and labels...")
        pkl_filename = os.path.join(self.data_path, f'x_embedded_{dataset_name}_{mode}.pkl')
        img_list_filename = os.path.join(self.data_path, f'{dataset_name}_image_batches_{mode}.pkl')

        x_embedded, y_train = joblib.load(pkl_filename)

        if x_embedded.shape[0] != y_train.shape[0]:
            min_samples = min(x_embedded.shape[0], y_train.shape[0])
            x_embedded = x_embedded[:min_samples]
            y_train = y_train[:min_samples]

        print(f'x_embedded shape: {x_embedded.shape}')
        print(f'y_train shape: {y_train.shape}')

        img_filenames = joblib.load(img_list_filename)
        images = []

        def process_images_in_chunks(img_filenames, chunk_size):
            for i in range(0, len(img_filenames), chunk_size):
                chunk_filenames = img_filenames[i:i + chunk_size]
                for img_filename in chunk_filenames:
                    loaded_images = joblib.load(img_filename)
                    images.extend(loaded_images)

        process_images_in_chunks(img_filenames, chunk_size=3)
        return x_embedded, y_train, images
