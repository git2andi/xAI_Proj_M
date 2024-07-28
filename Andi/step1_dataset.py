import numpy as np
import os
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import random
from torchvision.datasets import CIFAR10, CIFAR100
from medmnist import DermaMNIST, BreastMNIST

def load_embeddings(dataset_name, split, subsample):
    filepath = f'database/{dataset_name}/{split}.npz'
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} does not exist.")
    data = np.load(filepath)
    embeddings = data['embeddings']
    labels = data['labels']
    
    if subsample:
        embeddings = embeddings[::10]
        labels = labels[::10]

    labels = labels.reshape(-1)
    return embeddings, labels

def analyze_dataset(dataset_name, subsample):
    splits = ['train', 'val', 'test'] if dataset_name in ['dermamnist', 'breastmnist'] else ['train', 'test']
    analysis = [f"Dataset: {dataset_name}"]
    
    total_samples = 0
    combined_labels = []
    for split in splits:
        embeddings, labels = load_embeddings(dataset_name, split, subsample)
        combined_labels.extend(labels)
        num_samples = embeddings.shape[0]
        total_samples += num_samples
        analysis.append(f"Number of {split} samples: {num_samples}")
        analysis.append(f"{split.capitalize()} embedding dimensions: {embeddings.shape[1]}")
        analysis.append(f"Number of unique {split} labels: {len(np.unique(labels))}")
        analysis.append(f"{split.capitalize()} labels distribution:")
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            analysis.append(f"  Label {label}: {count} samples")

    analysis.insert(1, f"Total number of samples: {total_samples}")
    return "\n".join(analysis), combined_labels


def plot_tsne(embeddings, labels, dataset_name, output_dir, subsample):
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=labels, palette="pastel", legend="full")
    plt.title(f"{dataset_name.upper()} - {'Subsampled' if subsample else 'Full'} t-SNE")
    tsne_output_path = f"{output_dir}/{dataset_name}_{'subsampled' if subsample else 'full'}_tsne.png"
    plt.savefig(tsne_output_path)
    print(f"Saved t-SNE plot to {tsne_output_path}")


def plot_image_map(images, labels, dataset_name, output_dir):
    plt.figure(figsize=(10, 10))  # Adjusted figure size for a 3x3 grid
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        index = random.randint(0, len(images) - 1)
        plt.imshow(images[index])
        plt.title(f"Index {index}, Label {labels[index]}", fontsize=10)
        plt.axis('off')
    image_map_output_path = f"{output_dir}/{dataset_name}_image_map.png"
    plt.savefig(image_map_output_path)
    print(f"Saved image map to {image_map_output_path}")
 
def plot_label_distribution(labels, dataset_name, output_dir):
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=unique, y=counts, hue=unique, palette="pastel", dodge=False)
    plt.title(f"{dataset_name.upper()} - Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Frequency")
    plt.legend([],[], frameon=False)  # Hide the legend
    distribution_output_path = f"{output_dir}/{dataset_name}_label_distribution.png"
    plt.savefig(distribution_output_path)
    print(f"Saved label distribution plot to {distribution_output_path}")


def main(dataset_name, subsample):
    analysis_results, combined_labels = analyze_dataset(dataset_name, subsample)

    output_dir = 'step1_results'
    os.makedirs(output_dir, exist_ok=True)

    analysis_output_path = f"{output_dir}/{dataset_name}_analysis.txt"
    with open(analysis_output_path, 'w') as file:
        file.write(analysis_results)
    print(f"Saved analysis results to {analysis_output_path}")

    embeddings, labels = load_embeddings(dataset_name, 'train', subsample)

    if dataset_name in ['cifar10', 'cifar100']:
        dataset = CIFAR10(root="data", train=True, download=True, transform=None) if dataset_name == 'cifar10' else CIFAR100(root="data", train=True, download=True, transform=None)
        images = [dataset[i][0] for i in range(len(dataset))]
    else:
        dataset = DermaMNIST(split='train', download=True, transform=None) if dataset_name == 'dermamnist' else BreastMNIST(split='train', download=True, transform=None)
        images = [dataset[i][0] for i in range(len(dataset))]
    
    if subsample:
        images = images[::10]

    #plot_tsne(embeddings, labels, dataset_name, output_dir, subsample)
    plot_image_map(images, labels, dataset_name, output_dir)
    plot_label_distribution(np.array(combined_labels), dataset_name, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze datasets and create t-SNE plots.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., cifar10, cifar100, dermamnist, breastmnist)')
    parser.add_argument('--subsample', action='store_true', help='Whether to subsample the data (consider every tenth value)')
    args = parser.parse_args()

    print(f"Starting analysis for dataset {args.dataset} with subsampling {'enabled' if args.subsample else 'disabled'}...")
    main(args.dataset, args.subsample)
    print("Analysis complete.")
