import os
import sys
import argparse
import random
import joblib
from torchvision import datasets, transforms

# Add the src directory to the sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset_manager import DatasetManager
from embedding_calculator import EmbeddingCalculator
from image_processor import ImageProcessor
from visualization import Visualization

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process and visualize CIFAR-10 embeddings.")
    parser.add_argument("--analyze", action="store_true", help="Analyze the dataset.")
    parser.add_argument("--calculate", action="store_true", help="Calculate and save embeddings.")
    parser.add_argument("--index", type=int, help="Index to analyze. If not provided, no images will be generated.")
    parser.add_argument("--full", action="store_true", help="Use the full dataset.")
    parser.add_argument("--subsampled", action="store_true", help="Use a subsampled dataset.")
    args = parser.parse_args()

    dataset_name = "cifar10"
    root_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'database')
    data_full_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'data_full')
    data_subsampled_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'data_subsampled')
    images_presentation_path = os.path.join(os.path.dirname(__file__), '..', 'images_Presentation')

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

    # Transformation for CIFAR10 images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load CIFAR-10 images
    print("Loading CIFAR-10 images...")
    train_c10 = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    # test_c10 = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)

    # Print the length of train_c10 to ensure it's loaded
    print(f"Length of train_c10 after loading: {len(train_c10)}")

    dataset_manager = DatasetManager(root_path)
    image_processor = ImageProcessor(data_path)
    visualization = Visualization()

    if args.analyze:
        for dataset_name in ["cifar10", "cifar100", "dermamnist", "breastmnist"]:
            if dataset_name in ["cifar10", "cifar100"]:
                for dataset_type in ['train', 'test']:
                    dataset_manager.analyze_dataset(dataset_name, dataset_type)
            else:
                for dataset_type in ['train', 'val', 'test']:
                    dataset_manager.analyze_dataset(dataset_name, dataset_type)

    embedding_calculator = EmbeddingCalculator(root_path, data_path)

    if args.calculate:
        # Call the function to calculate and save full or subsampled embeddings
        embedding_calculator.calculate_and_save_embeddings(dataset_name, train_c10, mode)

    # Load and process images
    x_embedded, y_train, cifar10_images = embedding_calculator.load_and_process_images(dataset_name, mode)

    if args.index is not None:
        interesting_indices = image_processor.find_interesting_indices(x_embedded, y_train)

        # Determine the index to analyze
        if args.index == -1:
            if not interesting_indices:
                raise ValueError("No interesting indices found.")
            idx = random.choice(interesting_indices)
        else:
            idx = args.index

        print(f"Analyzing index: {idx}")
        idx, nearest_indices = image_processor.analyze_interesting_point(x_embedded, y_train, cifar10_images, interesting_indices, idx)

        # Print shapes of images to be plotted
        for i, image in enumerate([cifar10_images[idx]] + [cifar10_images[i] for i in nearest_indices]):
            print(f"Image {i} shape before plotting: {image.shape}")

        images = [cifar10_images[idx]] + [cifar10_images[i] for i in nearest_indices]
        titles = [f"Original: Label {y_train[idx]}"] + [f"Neighbor: Label {y_train[i]}" for i in nearest_indices]
        heading = f'Nearest Neighbors for Index {idx} (Label {y_train[idx]})'

        images_save_path = os.path.join(images_presentation_path, f'index{idx}_{mode}_images.png')
        tsne_save_path = os.path.join(images_presentation_path, f'index{idx}_{mode}_tsne.png')

        print(f"Plotting images with titles: {titles}")
        visualization.plot_images(images, titles, heading, nrows=1, ncols=6, save_path=images_save_path)
        print("Plotting t-SNE...")
        visualization.plot_tsne(x_embedded, y_train, idx, save_path=tsne_save_path)
    else:
        print("No index provided. Skipping image generation.")
        # Directly plot t-SNE without highlighting a specific index
        tsne_save_path = os.path.join(images_presentation_path, f'index_none_{mode}_tsne.png')
        visualization.plot_tsne(x_embedded, y_train, save_path=tsne_save_path)

if __name__ == "__main__":
    main()
