import os
import sys
import argparse
import random
import joblib
from torchvision import datasets, transforms
from medmnist import DermaMNIST, BreastMNIST

# Add the src directory to the sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dataset_manager import DatasetManager
from embedding_calculator import EmbeddingCalculator
from image_processor import ImageProcessor
from visualization import Visualization

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process and visualize dataset embeddings.")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100", "dermamnist", "breastmnist"], help="Specify the dataset to use.")
    parser.add_argument("--analyze", action="store_true", help="Analyze the dataset.")
    parser.add_argument("--calculate", action="store_true", help="Calculate and save embeddings.")
    parser.add_argument("--index", type=int, help="Index to analyze. If not provided, no images will be generated.")
    parser.add_argument("--full", action="store_true", help="Use the full dataset.")
    parser.add_argument("--subsampled", action="store_true", help="Use a subsampled dataset.")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Define paths
    root_path = os.path.join(os.path.dirname(__file__), 'data', 'database')
    data_full_path = os.path.join(os.path.dirname(__file__), 'data', 'data_full', args.dataset)
    data_subsampled_path = os.path.join(os.path.dirname(__file__), 'data', 'data_subsampled', args.dataset)
    images_presentation_path = os.path.join(os.path.dirname(__file__), 'images_Presentation', args.dataset)

    # Ensure the images_Presentation path exists
    os.makedirs(images_presentation_path, exist_ok=True)

    # Determine which dataset to use
    if args.full:
        data_path = data_full_path
        mode = "full"
    elif args.subsampled:
        data_path = data_subsampled_path
        mode = "subsampled"
    else:
        raise ValueError("Please specify either --full or --subsampled")

    # Ensure save paths exist
    os.makedirs(data_path, exist_ok=True)

    # Transformation for images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load dataset images
    if args.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    elif args.dataset == "cifar100":
        train_dataset = datasets.CIFAR100(root='data', train=True, download=True, transform=transform)
    elif args.dataset == "dermamnist":
        train_dataset = DermaMNIST(split="train", download=True, transform=transform)
    elif args.dataset == "breastmnist":
        train_dataset = BreastMNIST(split="train", download=True, transform=transform)

    print(f"Length of {args.dataset} after loading: {len(train_dataset)}")

    dataset_manager = DatasetManager(root_path)
    image_processor = ImageProcessor(data_path)
    visualization = Visualization()
    embedding_calculator = EmbeddingCalculator(root_path, data_path)

    if args.analyze:
        analyze_datasets(dataset_manager, args.dataset)

    if args.calculate:
        embedding_calculator.calculate_and_save_embeddings(args.dataset, train_dataset, mode)

    x_embedded, y_train, images = embedding_calculator.load_and_process_images(args.dataset, mode)

    if args.index is not None:
        process_index(args, image_processor, visualization, x_embedded, y_train, images, images_presentation_path, mode)
    else:
        tsne_save_path = os.path.join(images_presentation_path, f'index_none_{mode}_tsne.png')
        visualization.plot_tsne(x_embedded, y_train, save_path=tsne_save_path)

def analyze_datasets(dataset_manager, dataset_name):
    """Analyze datasets."""
    dataset_types = ['train', 'test'] if dataset_name in ["cifar10", "cifar100"] else ['train', 'val', 'test']
    for dataset_type in dataset_types:
        dataset_manager.analyze_dataset(dataset_name, dataset_type)

def process_index(args, image_processor, visualization, x_embedded, y_train, images, images_presentation_path, mode):
    """Process and visualize a specific index."""
    interesting_indices = image_processor.find_interesting_indices(x_embedded, y_train)

    idx = random.choice(interesting_indices) if args.index == -1 else args.index
    print(f"Analyzing index: {idx}")

    idx, nearest_indices = image_processor.analyze_interesting_point(x_embedded, y_train, images, interesting_indices, idx)

    images_to_plot = [images[idx]] + [images[i] for i in nearest_indices]
    titles = [f"Original: Label {y_train[idx]}"] + [f"Neighbor: Label {y_train[i]}" for i in nearest_indices]
    heading = f'Nearest Neighbors for Index {idx} (Label {y_train[idx]})'

    images_save_path = os.path.join(images_presentation_path, f'index{idx}_{mode}_images.png')
    tsne_save_path = os.path.join(images_presentation_path, f'index{idx}_{mode}_tsne.png')

    visualization.plot_images(images_to_plot, titles, heading, nrows=1, ncols=6, save_path=images_save_path)
    visualization.plot_tsne(x_embedded, y_train, idx, save_path=tsne_save_path)

if __name__ == "__main__":
    main()
