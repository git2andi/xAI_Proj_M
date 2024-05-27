
# CIFAR-10 Embedding and Visualization Project

This project is designed to process and visualize embeddings of the CIFAR-10 dataset. The project includes functionalities for analyzing datasets, calculating and saving embeddings using t-SNE, and visualizing the embeddings along with their nearest neighbors in the dataset.

## Project Structure

- `src/`: Contains all source code files.
  - `main.py`: The main script to run the project.
  - `dataset_manager.py`: Handles dataset loading and analysis.
  - `embedding_calculator.py`: Calculates and saves embeddings.
  - `image_processor.py`: Processes image batches and finds interesting indices.
  - `visualization.py`: Contains methods for visualizing images and t-SNE embeddings.
- `data/`: Directory for storing dataset files and processed embeddings.

## Setup and Installation

1. **Clone the Repository:**
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

## Usage

### Command Line Arguments

- `--analyze`: Analyze the dataset.
- `--calculate`: Calculate and save embeddings.
- `--index <int>`: Index to analyze. Use -1 for a random index.
- Example: Analyze a Specific or Random Index:
   ```
   python src/main.py --index 42  # Specific index
   python src/main.py --index -1  # Random index from interesting points
   ```

## Notes

- Ensure that the dataset is available in the specified paths.
- The image processing steps ensure that the images are correctly reshaped and normalized before visualization.
- The embeddings and image batches are saved to avoid recomputation on subsequent runs.


## License

This project is licensed under the MIT License. See the LICENSE file for more details.
