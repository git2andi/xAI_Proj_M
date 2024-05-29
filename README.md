
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


## Usage
### Dataset anaylsis
1. Run the script:

```sh
python main.py --dataset cifar10
python main.py --dataset cifar100
python main.py --dataset dermamnist
python main.py --dataset breastmnist

```

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


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
