import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from torchvision import datasets, transforms
from medmnist import DermaMNIST, BreastMNIST
from PIL import Image
import argparse

root_path = "./database"
datasets_dict = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "dermamnist": DermaMNIST,
    "breastmnist": BreastMNIST
}

def load_embeddings_and_labels(dataset_name, root_path, subsample=True):
    data = np.load(os.path.join(root_path, dataset_name, 'train.npz'))
    X, y = data['embeddings'], data['labels'].reshape(-1,)
    if subsample:
        X, y = X[::10], y[::10]
    return X, y

def load_images(dataset_name, subsample=True):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    if dataset_name in ["cifar10", "cifar100"]:
        train_dataset = datasets_dict[dataset_name](root='data', train=True, download=True, transform=transform)
    else:
        train_dataset = datasets_dict[dataset_name](split="train", download=True, transform=transform)
    
    if subsample:
        indices = np.arange(0, len(train_dataset), 10)
        subsampled_dataset = [train_dataset[i] for i in indices]
        return subsampled_dataset
    return train_dataset

def visualize_tsne(X, y, dataset_name, selected_index, mode, random_state=42):
    x_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3, random_state=random_state).fit_transform(X)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=x_embedded[:, 0], y=x_embedded[:, 1], hue=y, palette="pastel")
    plt.scatter(x_embedded[selected_index, 0], x_embedded[selected_index, 1], color='red', s=100, edgecolor='black', label=f'Index {selected_index}')
    plt.annotate(f'Index {selected_index}', (x_embedded[selected_index, 0], x_embedded[selected_index, 1]), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.title('t-SNE Visualization')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    
    image_dir = "images"
    os.makedirs(image_dir, exist_ok=True)
    tsne_filename = os.path.join(image_dir, f'{dataset_name}_index{selected_index}_{mode}_tsne.png')
    plt.savefig(tsne_filename)
    plt.show()
    
    return x_embedded

def find_interesting_points(X, y, min_neighbors=1, max_neighbors=5, threshold=10):
    interesting_points = []
    for num_neighbors in range(max_neighbors, min_neighbors - 1, -1):
        nbrs = NearestNeighbors(n_neighbors=num_neighbors + 1).fit(X)
        distances, indices = nbrs.kneighbors(X)
        for i in range(len(X)):
            neighbor_labels = y[indices[i][1:]]  # Skip the first neighbor as it's the point itself
            unique_labels = np.unique(neighbor_labels)
            if len(unique_labels) >= num_neighbors:
                interesting_points.append(i)
        print(f"Found {len(interesting_points)} interesting points with at least {num_neighbors} different labels.")
        if len(interesting_points) >= threshold:
            break
    return interesting_points

def display_nearest_neighbors(X, y, train_dataset, x_embedded, selected_index, mode, dataset_name):
    distances = euclidean_distances([x_embedded[selected_index]], x_embedded)[0]
    nearest_indices = np.argsort(distances)[1:6]
    
    images_to_plot = [train_dataset[selected_index][0].numpy()] + [train_dataset[i][0].numpy() for i in nearest_indices]
    titles = [f"Original: Label {y[selected_index]}"] + [f"Neighbor: Label {y[i]}" for i in nearest_indices]

    fig, axes = plt.subplots(1, 6, figsize=(15, 5))
    for ax, img, title in zip(axes, images_to_plot, titles):
        img = img.transpose((1, 2, 0))  # Change shape from (C, H, W) to (H, W, C)
        img = (img * 255).astype(np.uint8)  # Convert from float tensor to uint8 for PIL
        
        if img.shape[2] == 1:  # Grayscale
            img = img.squeeze(axis=2)
            pil_img = Image.fromarray(img, mode='L')
        else:  # RGB
            pil_img = Image.fromarray(img)
        
        ax.imshow(pil_img, cmap='gray' if img.ndim == 2 else None)
        ax.set_title(title)
        ax.axis('off')
    
    plt.suptitle(f'Nearest Neighbors for Index {selected_index} (Label {y[selected_index]})')
    plt.tight_layout()

    image_dir = "images"
    os.makedirs(image_dir, exist_ok=True)
    images_filename = os.path.join(image_dir, f'{dataset_name}_index{selected_index}_{mode}_images.png')
    plt.savefig(images_filename)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize dataset embeddings with t-SNE")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100", "dermamnist", "breastmnist"], help="Dataset to use")
    parser.add_argument("--subsample", action="store_true", help="Use subsampled dataset")
    parser.add_argument("--full", action="store_true", help="Use full dataset")
    args = parser.parse_args()
    
    dataset_name = args.dataset
    subsample = args.subsample

    if args.full and args.subsample:
        raise ValueError("Specify either --subsample or --full, not both.")

    mode = "subsampled" if subsample else "full"
    
    X, y = load_embeddings_and_labels(dataset_name, root_path, subsample)
    train_dataset = load_images(dataset_name, subsample)

    interesting_points = find_interesting_points(X, y)
    selected_index = np.random.choice(interesting_points) # or seelct own index
    
    display_nearest_neighbors(X, y, train_dataset, X, selected_index, mode, dataset_name)
    x_embedded = visualize_tsne(X, y, dataset_name, selected_index, mode)

if __name__ == "__main__":
    main()
