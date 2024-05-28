
# xAI Project

This project explores the latent space of different datasets using t-SNE, evaluates the performance of a k-Nearest Neighbour (k-NN) classifier, and trains a Linear Probing model on the embeddings. The project consists of two main parts: k-NN classification and Linear Probing. The results are summarized and saved for further analysis.

## Project Structure

```
.
├── database
│   ├── cifar10
│   │   ├── train.npz
│   │   ├── test.npz
│   ├── cifar100
│   │   ├── train.npz
│   │   ├── test.npz
│   ├── dermamnist
│   │   ├── train.npz
│   │   ├── test.npz
│   ├── breastmnist
│       ├── train.npz
│       ├── test.npz
├── images
│   ├── (generated images will be saved here)
├── knn.py
├── linear_probing.py
├── requirements.txt
├── README.md
```

## Prerequisites

- Python 3.6 or higher
- CUDA-enabled GPU (optional, for faster computations)

## Installation

1. Clone the repository:

```sh
git clone https://github.com/your-username/xAI_Project.git
cd xAI_Project
```

2. Create a virtual environment and activate it:

```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required packages:

```sh
pip install -r requirements.txt
```

## Usage

### k-Nearest Neighbour (k-NN) Classification

1. Run the k-NN script:

```sh
python knn.py --dataset cifar10
python knn.py --dataset breastmnist
```

2. The results will be saved in a text file named according to the dataset, e.g., `cifar10_results.txt`.

### Linear Probing

1. Run the Linear Probing script:

```sh
python linear_probing.py --dataset cifar10
python linear_probing.py --dataset breastmnist
```

2. The results will be saved in a text file named according to the dataset, e.g., `cifar10_linear_probing_results.txt`.

## Scripts

### knn.py

This script performs k-Nearest Neighbour (k-NN) classification on the embeddings of the specified dataset. It uses 15-fold cross-validation to determine the best `k` value and evaluates the model on the test set.

### linear_probing.py

This script trains a Linear Probing model on the embeddings of the specified dataset. It evaluates different hyperparameters (learning rate, batch size, number of epochs) to find the best configuration and evaluates the model on the test set.

## Requirements

The `requirements.txt` file contains the list of required packages:

```txt
numpy
torch
scikit-learn
matplotlib
seaborn
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Professor's tutorial on t-SNE and k-NN.
- PyTorch documentation and tutorials.
- Scikit-learn documentation.

---

By following the instructions in this README, you should be able to set up the project environment, run the scripts, and analyze the results of k-NN classification and Linear Probing on various datasets.
