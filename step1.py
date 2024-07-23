import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from torchvision.datasets import CIFAR10, CIFAR100
from medmnist import DermaMNIST, BreastMNIST
import os
import argparse
import seaborn as sns
import random

def load_embeddings(dataset_name):
    filepath = f'database/{dataset_name}/train.npz'
    print(f"Loading embeddings from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} does not exist.")
    data = np.load(filepath)
    return data['embeddings'], data['labels']

def plot_tsne(embeddings, labels, title, selected_index=None, nearest_indices=None, furthest_indices=None, output_path=None):
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=labels, palette="pastel", legend="full")

    if nearest_indices is not None:
        for idx in nearest_indices:
            plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], color='lightgreen', edgecolor='black', s=100, label=f'Nearest {idx}')
            plt.text(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], f'{idx}', color='black')

    if furthest_indices is not None:
        for idx in furthest_indices:
            plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], color='violet', edgecolor='black', s=100, label=f'Furthest {idx}')
            plt.text(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], f'{idx}', color='black')

    if selected_index is not None:
        plt.scatter(reduced_embeddings[selected_index, 0], reduced_embeddings[selected_index, 1], color='red', s=100, label=f'Index {selected_index}')
        plt.text(reduced_embeddings[selected_index, 0], reduced_embeddings[selected_index, 1], f'Index {selected_index}', color='black')

    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.title(title)
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved t-SNE plot to {output_path}")


def find_nearest_neighbors(embeddings, n_neighbors=6):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    indices = indices[:, 1:]  # Exclude first neighbor
    return distances, indices

def find_furthest_neighbors(embeddings, n_neighbors=5):
    nbrs = NearestNeighbors(n_neighbors=embeddings.shape[0], algorithm='ball_tree').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    furthest_indices = indices[:, -n_neighbors:]
    return distances, furthest_indices

def plot_combined_neighbors(images, labels, selected_index, nearest_indices, furthest_indices, title='', output_path=None):
    print(f"Plotting combined neighbors for target index {selected_index}...")
    fig, ax = plt.subplots(2, 6, figsize=(18, 6))

    # top row
    ax[0, 0].imshow(images[selected_index])
    ax[0, 0].set_title(f"Selected Index {selected_index}\nLabel {labels[selected_index]}", fontsize=12)
    ax[0, 0].axis('off')
    for i in range(5):
        ax[0, i + 1].imshow(images[nearest_indices[i]])
        ax[0, i + 1].set_title(f"Index {nearest_indices[i]}\nLabel {labels[nearest_indices[i]]}", fontsize=10)
        ax[0, i + 1].axis('off')

    # bottom row
    ax[1, 0].imshow(images[selected_index])
    ax[1, 0].set_title(f"Selected Index {selected_index}\nLabel {labels[selected_index]}", fontsize=12)
    ax[1, 0].axis('off')
    for i in range(5):
        ax[1, i + 1].imshow(images[furthest_indices[i]])
        ax[1, i + 1].set_title(f"Index {furthest_indices[i]}\nLabel {labels[furthest_indices[i]]}", fontsize=10)
        ax[1, i + 1].axis('off')

    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.85, hspace=0.6)
    if output_path:
        plt.savefig(output_path)
        print(f"Saved combined neighbors plot to {output_path}")

def main(dataset_name, subsample):
    datasets = {
        'cifar10': CIFAR10,
        'cifar100': CIFAR100,
        'dermamnist': DermaMNIST,
        'breastmnist': BreastMNIST
    }

    embeddings, labels = load_embeddings(dataset_name)
    
    if subsample:
        embeddings, labels = embeddings[::10], labels[::10].reshape(-1,)
    else:
        labels = labels.reshape(-1,)

    output_dir = 'step1_results'
    os.makedirs(output_dir, exist_ok=True)

    distances, nearest_indices = find_nearest_neighbors(embeddings)
    _, furthest_indices = find_furthest_neighbors(embeddings)
    print(f"Loading images for {dataset_name}...")
    if dataset_name in ['cifar10', 'cifar100']:
        dataset = datasets[dataset_name](root="data", train=True, download=True, transform=None)
        images = [dataset[i][0] for i in range(len(dataset))]
    else:
        dataset = datasets[dataset_name](split='train', download=True, transform=None)
        images = [dataset[i][0] for i in range(len(dataset))]
    
    if subsample:
        images = images[::10]
    
    # Find interesting points
    interesting_indices = {4: [], 3: [], 2: []}
    for i in range(len(labels)):
        neighbor_labels = labels[nearest_indices[i, 1:]]
        unique_labels = len(set(neighbor_labels))
        if unique_labels in interesting_indices:
            interesting_indices[unique_labels].append(i)
    
    selected_index = None
    for num_labels in [4, 3, 2]:
        if interesting_indices[num_labels]:
            selected_index = random.choice(interesting_indices[num_labels])
            print(f"Random interesting point index: {selected_index} with {num_labels} different label neighbors")
            break

    if selected_index is not None:
        tsne_output_path = f"{output_dir}/{dataset_name}_index{selected_index}_{'subsampled' if subsample else 'full'}_tsne.png"
        plot_tsne(embeddings, labels, f'{dataset_name.upper()} - Train Split', selected_index=selected_index, nearest_indices=nearest_indices[selected_index], furthest_indices=furthest_indices[selected_index], output_path=tsne_output_path)
        
        combined_output_path = f"{output_dir}/{dataset_name}_index{selected_index}_{'subsampled' if subsample else 'full'}_neighbors.png"
        plot_combined_neighbors(
            images, labels, selected_index, 
            nearest_indices[selected_index], furthest_indices[selected_index],
            title=f'Neighbors for Index {selected_index} (Label {labels[selected_index]})',
            output_path=combined_output_path
        )
    else:
        print("No interesting points found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze embeddings for a given dataset with optional subsampling.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., cifar10, cifar100, dermamnist, breastmnist)')
    parser.add_argument('--subsample', action='store_true', help='Whether to subsample the data (consider every tenth value)')
    args = parser.parse_args()
    
    print(f"Starting analysis for dataset {args.dataset} with subsampling {'enabled' if args.subsample else 'disabled'}...")
    main(args.dataset, args.subsample)
    print("Analysis complete.")
