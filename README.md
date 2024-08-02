
# xAI Project

This project explores the latent space of different datasets using t-SNE, evaluates the performance of a k-Nearest Neighbour (k-NN) classifier, and trains a Linear Probing model on the embeddings. The project consists of two main parts: k-NN classification and Linear Probing. The results are summarized and saved for further analysis.


## Usage
### Step 1
1. Run the script:

```sh
python step1.py --dataset cifar10 # cifar100, dermamnist, breastmnist
python step1.py --dataset cifar10 --subsample  # Add if required 
python step1_dataset.py --dataset cifar10
```

### k-Nearest Neighbour (k-NN) Classification vs Linear Layer

1. Run the k-NN script:

```sh
python step2.py --dataset cifar10   # no subsample here possible
```

### ForestKNN

1. Run the k-NN script:

```sh
python step3_v1.py --dataset cifar10   # no subsample aswell; adapt version
```


2. The results will be saved in the respective step1, step2 or step3 results folder


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
