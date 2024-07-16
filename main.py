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
from mpl_toolkits.mplot3d import Axes3D
import umap.umap_ as umap
import pandas as pd

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


def visualize_tsne_3d(X, y, dataset_name, selected_index, mode, random_state=42):
    x_embedded = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3, random_state=random_state).fit_transform(X)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x_embedded[:, 0], x_embedded[:, 1], x_embedded[:, 2], c=y, cmap="viridis", s=50)
    ax.scatter(x_embedded[selected_index, 0], x_embedded[selected_index, 1], x_embedded[selected_index, 2], color='red', s=100, edgecolor='black', label=f'Index {selected_index}')
    plt.title('3D t-SNE Visualization')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    image_dir = "images"
    os.makedirs(image_dir, exist_ok=True)
    tsne_filename = os.path.join(image_dir, f'{dataset_name}_index{selected_index}_{mode}_tsne_3d.png')
    plt.savefig(tsne_filename)
    
    plt.show()
    
    return x_embedded


def visualize_umap(X, y, dataset_name, selected_index, mode):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=y, palette="pastel")
    plt.scatter(embedding[selected_index, 0], embedding[selected_index, 1], color='red', s=100, edgecolor='black', label=f'Index {selected_index}')
    plt.title('UMAP Visualization')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    image_dir = "images"
    os.makedirs(image_dir, exist_ok=True)
    umap_filename = os.path.join(image_dir, f'{dataset_name}_index{selected_index}_{mode}_umap.png')
    plt.savefig(umap_filename)

    plt.show()
    
    return embedding


def visualize_tsne_pairplot(X, y, dataset_name, selected_index, mode, random_state=42):
    x_embedded = TSNE(n_components=4, learning_rate='auto', init='random', perplexity=3, random_state=random_state, method='exact').fit_transform(X)
    df = pd.DataFrame(x_embedded, columns=['Dim1', 'Dim2', 'Dim3', 'Dim4'])
    df['label'] = y
    
    sns.pairplot(df, hue='label', palette='pastel')
    plt.suptitle('Pairplot of t-SNE Dimensions')

    image_dir = "images"
    os.makedirs(image_dir, exist_ok=True)
    pairplot_filename = os.path.join(image_dir, f'{dataset_name}_index{selected_index}_{mode}_tsne_pairplot.png')
    plt.savefig(pairplot_filename)

    plt.show()
    
    return x_embedded


def visualize_distance_heatmap(X, y, dataset_name, selected_index, mode):
    distances = euclidean_distances(X)
    plt.figure(figsize=(12, 10))
    sns.heatmap(distances, cmap='viridis')
    plt.title('Distance Matrix Heatmap')

    image_dir = "images"
    os.makedirs(image_dir, exist_ok=True)
    heatmap_filename = os.path.join(image_dir, f'{dataset_name}_index{selected_index}_{mode}_distance_heatmap.png')
    plt.savefig(heatmap_filename)

    plt.show()



def visualize_embeddings(X, y, dataset_name, selected_index, mode, method='tsne', random_state=42):
    if method == 'tsne':
        return visualize_tsne(X, y, dataset_name, selected_index, mode, random_state)
    elif method == 'tsne_3d':
        return visualize_tsne_3d(X, y, dataset_name, selected_index, mode, random_state)
    elif method == 'umap':
        return visualize_umap(X, y, dataset_name, selected_index, mode)
    elif method == 'pairplot':
        return visualize_tsne_pairplot(X, y, dataset_name, selected_index, mode, random_state)
    elif method == 'heatmap':
        return visualize_distance_heatmap(X, y, dataset_name, selected_index, mode)
    else:
        raise ValueError(f"Unknown visualization method: {method}")


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

def display_neighbors(X, y, train_dataset, x_embedded, selected_index, mode, dataset_name, num_neighbors=5, num_far_points=5):
    distances = euclidean_distances([x_embedded[selected_index]], x_embedded)[0]
    nearest_indices = np.argsort(distances)[1:num_neighbors+1]
    farthest_indices = np.argsort(distances)[-num_far_points:]

    images_to_plot = [train_dataset[selected_index][0].numpy()] + [train_dataset[i][0].numpy() for i in nearest_indices] + [train_dataset[i][0].numpy() for i in farthest_indices]
    titles = [f"Original: Label {y[selected_index]}"] + [f"Neighbor: Label {y[i]}" for i in nearest_indices] + [f"Far: Label {y[i]}" for i in farthest_indices]

    fig, axes = plt.subplots(2, max(num_neighbors, num_far_points) + 1, figsize=(15, 10))

    # Plot original image on the left side of both rows
    for row in range(2):
        img = images_to_plot[0].transpose((1, 2, 0))  # Change shape from (C, H, W) to (H, W, C)
        img = (img * 255).astype(np.uint8)  # Convert from float tensor to uint8 for PIL
        
        if img.shape[2] == 1:  # Grayscale
            img = img.squeeze(axis=2)
            pil_img = Image.fromarray(img, mode='L')
        else:  # RGB
            pil_img = Image.fromarray(img)
        
        axes[row, 0].imshow(pil_img, cmap='gray' if img.ndim == 2 else None)
        axes[row, 0].set_title(titles[0])
        axes[row, 0].axis('off')

    # Plot nearest neighbors on the first row
    for idx, (ax, img, title) in enumerate(zip(axes[0, 1:], images_to_plot[1:num_neighbors+1], titles[1:num_neighbors+1])):
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

    # Plot farthest neighbors on the second row
    for idx, (ax, img, title) in enumerate(zip(axes[1, 1:], images_to_plot[num_neighbors+1:], titles[num_neighbors+1:])):
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

    plt.suptitle(f'Neighbors for Index {selected_index} (Label {y[selected_index]})')
    plt.tight_layout()

    image_dir = "images"
    os.makedirs(image_dir, exist_ok=True)
    images_filename = os.path.join(image_dir, f'{dataset_name}_index{selected_index}_{mode}_neighbors_far.png')
    plt.savefig(images_filename)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize dataset embeddings with t-SNE")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100", "dermamnist", "breastmnist"], help="Dataset to use")
    parser.add_argument("--subsample", action="store_true", help="Use subsampled dataset")
    parser.add_argument("--full", action="store_true", help="Use full dataset")
    parser.add_argument("--visualization", type=str, default="tsne", choices=["tsne", "tsne_3d", "umap", "pairplot", "heatmap"], help="Visualization method")
    args = parser.parse_args()
    
    dataset_name = args.dataset
    subsample = args.subsample

    if args.full and args.subsample:
        raise ValueError("Specify either --subsample or --full, not both.")

    mode = "subsampled" if subsample else "full"
    
    X, y = load_embeddings_and_labels(dataset_name, root_path, subsample)
    train_dataset = load_images(dataset_name, subsample)

    interesting_points = find_interesting_points(X, y)
    selected_index = np.random.choice(interesting_points) # or select your own index
    
    display_neighbors(X, y, train_dataset, X, selected_index, mode, dataset_name)
    x_embedded = visualize_embeddings(X, y, dataset_name, selected_index, mode, method=args.visualization)

if __name__ == "__main__":
    main()

