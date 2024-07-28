import os
import numpy as np
import torch
import argparse
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ForestKNN; play with various distances

root_path = "./database"

class ForestKNN:
    """
    Forest of k-Nearest Neighbors (kNN) classifiers.
    """
    def __init__(self, components=10, k=5, sample_size=0.8, feature_size=0.8, distance_metric='euclidean', device='cuda'):
        self.components = components
        self.k = k
        self.sample_size = sample_size
        self.feature_size = feature_size
        self.distance_metric = distance_metric
        self.device = device
        self.trees = []
        self.feature_indices = []

    def fit(self, X, y):
        """
        Fits each kNN model on a random subset of the training data and features.
        """
        num_samples = int(self.sample_size * X.shape[0])
        num_features = int(self.feature_size * X.shape[1])

        for i in range(self.components):
            sample_indices = np.random.choice(X.shape[0], num_samples, replace=True)
            feature_indices = np.random.choice(X.shape[1], num_features, replace=True)
            self.feature_indices.append(feature_indices)

            X_subset = X[sample_indices][:, feature_indices]
            y_subset = y[sample_indices]

            tree = self._fit_knn_model(X_subset, y_subset, self.k)
            self.trees.append(tree)

    def _fit_knn_model(self, X, y, k):
        """
        Train the kNN model with the specified number of neighbors (k).
        """
        X_train = torch.tensor(X, device=self.device)
        y_train = torch.tensor(y, device=self.device)
        return (X_train, y_train)

    def _compute_distances(self, X, X_train):
        """
        Compute distances between X and X_train based on the specified metric.
        """
        if self.distance_metric == 'euclidean':
            return torch.cdist(X, X_train, p=2)
        elif self.distance_metric == 'manhattan':
            return torch.cdist(X, X_train, p=1)
        elif self.distance_metric == 'cosine':
            X_norm = torch.nn.functional.normalize(X, p=2, dim=1)
            X_train_norm = torch.nn.functional.normalize(X_train, p=2, dim=1)
            return 1 - torch.mm(X_norm, X_train_norm.T)
        elif self.distance_metric == 'chebyshev':
            return torch.cdist(X, X_train, p=float('inf'))
        elif self.distance_metric == 'minkowski':
            p = 3
            return torch.cdist(X, X_train, p=p)
        elif self.distance_metric == 'mahalanobis':
            # identity matrix is inverse covariance matrix
            VI = torch.eye(X.shape[1], device=self.device)
            delta = X.unsqueeze(1) - X_train.unsqueeze(0)
            return torch.sqrt(torch.sum(delta @ VI * delta, dim=-1))
        elif self.distance_metric == 'hamming':
            return (X.unsqueeze(1) != X_train.unsqueeze(0)).float().mean(dim=2)
        elif self.distance_metric == 'canberra':
            delta = torch.abs(X.unsqueeze(1) - X_train.unsqueeze(0))
            sum_vals = torch.abs(X.unsqueeze(1)) + torch.abs(X_train.unsqueeze(0))
            return torch.sum(delta / sum_vals, dim=2)
        elif self.distance_metric == 'braycurtis':
            num = torch.sum(torch.abs(X.unsqueeze(1) - X_train.unsqueeze(0)), dim=2)
            denom = torch.sum(torch.abs(X.unsqueeze(1) + X_train.unsqueeze(0)), dim=2)
            return num / denom
        elif self.distance_metric == 'jaccard':
            intersection = torch.min(X.unsqueeze(1), X_train.unsqueeze(0)).sum(dim=2)
            union = torch.max(X.unsqueeze(1), X_train.unsqueeze(0)).sum(dim=2)
            return 1 - intersection / union
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")


    def _predict_knn(self, X, X_train, y_train, k):
        """
        Predicts the labels for the input data using kNN.
        """
        X = torch.tensor(X, device=self.device)
        distances = self._compute_distances(X, X_train)
        neighbors = distances.argsort(dim=1)[:, :k]
        top_labels = y_train[neighbors]
        predictions = torch.mode(top_labels, dim=1).values
        return predictions.cpu().numpy()

    def predict(self, X):
        """
        Predicts the labels by aggregating the predictions of each kNN model.
        """
        predictions = np.zeros((X.shape[0], self.components))

        for i, (X_train, y_train) in enumerate(self.trees):
            X_subset = X[:, self.feature_indices[i]]
            predictions[:, i] = self._predict_knn(X_subset, X_train, y_train, self.k)

        final_predictions = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=predictions)
        return final_predictions

def load_embeddings_and_labels(dataset_name, root_path):
    """
    Load embeddings and labels for the specified dataset.
    """
    if dataset_name in ["dermamnist", "breastmnist"]:
        train_data = np.load(os.path.join(root_path, dataset_name, 'train.npz'))
        val_data = np.load(os.path.join(root_path, dataset_name, 'val.npz'))
        test_data = np.load(os.path.join(root_path, dataset_name, 'test.npz'))
        X_train, y_train = train_data['embeddings'], train_data['labels'].reshape(-1,)
        X_val, y_val = val_data['embeddings'], val_data['labels'].reshape(-1,)
        X_test, y_test = test_data['embeddings'], test_data['labels'].reshape(-1,)
    else:
        train_data = np.load(os.path.join(root_path, dataset_name, 'train.npz'))
        test_data = np.load(os.path.join(root_path, dataset_name, 'test.npz'))
        X_train, y_train = train_data['embeddings'], train_data['labels'].reshape(-1,)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=None)
        X_test, y_test = test_data['embeddings'], test_data['labels'].reshape(-1,)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate_knn_performance(y_pred, y_true):
    """
    Evaluate the performance of the kNN model using various metrics.
    """
    accuracy = accuracy_score(y_pred, y_true)
    precision = precision_score(y_pred, y_true, average='weighted', zero_division=0)
    recall = recall_score(y_pred, y_true, average='weighted', zero_division=0)
    f1 = f1_score(y_pred, y_true, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

def main(dataset_name):
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_embeddings_and_labels(dataset_name, root_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # derma and breast run fast with full config
    num_trees_list = [5, 10, 20, 25]
    k_list = [3, 5, 10, 15]
    sample_size_list = [0.6, 0.8, 1.0]
    feature_size_list = [0.6, 0.8, 1.0]
    distance_metric_list = ['euclidean', 'manhattan', 'cosine', 'chebyshev', 'minkowski', 'mahalanobis', 'hamming', 'canberra', 'braycurtis', 'jaccard']
    
    # for Cifar use reduced parameters
    #num_trees_list = [5, 10]
    #k_list = [3, 10, 15]
    #sample_size_list = [0.6, 0.8]
    #feature_size_list = [0.6, 0.8]
    #distance_metric_list = ['euclidean', 'cosine'] 


    best_val_accuracy = 0
    best_params = None

    output_file = f"{dataset_name}_forestknn_step3_v4.txt"
    with open(output_file, "w") as f:
        f.write("Results:\n")

        for num_trees, k, sample_size, feature_size, distance_metric in product(num_trees_list, k_list, sample_size_list, feature_size_list, distance_metric_list):
            forest_knn = ForestKNN(components=num_trees, k=k, sample_size=sample_size, feature_size=feature_size, distance_metric=distance_metric, device=device)
            forest_knn.fit(X_train, y_train)

            y_val_pred = forest_knn.predict(X_val)
            val_accuracy, val_precision, val_recall, val_f1 = evaluate_knn_performance(y_val_pred, y_val)
            result_str = f'Validation - Trees: {num_trees}, k: {k}, Sample Size: {sample_size}, Feature Size: {feature_size}, Distance: {distance_metric}, Accuracy: {val_accuracy:.4f}'
            print(result_str)
            f.write(result_str + "\n")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_params = (num_trees, k, sample_size, feature_size, distance_metric)

        best_params_str = f'Best Validation Parameters - Trees: {best_params[0]}, k: {best_params[1]}, Sample Size: {best_params[2]}, Feature Size: {best_params[3]}, Distance: {best_params[4]}, Accuracy: {best_val_accuracy:.4f}'
        print(best_params_str)

        forest_knn = ForestKNN(components=best_params[0], k=best_params[1], sample_size=best_params[2], feature_size=best_params[3], distance_metric=best_params[4], device=device)
        forest_knn.fit(X_train, y_train)
        y_test_pred = forest_knn.predict(X_test)
        test_accuracy, test_precision, test_recall, test_f1 = evaluate_knn_performance(y_test_pred, y_test)
        test_results_str = f'Test Accuracy: {test_accuracy:.4f}\nTest Precision: {test_precision:.4f}\nTest Recall: {test_recall:.4f}\nTest F1 Score: {test_f1:.4f}'
        print(test_results_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forest k-NN with PyTorch")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100", "dermamnist", "breastmnist"], help="Dataset to use")
    args = parser.parse_args()

    dataset_name = args.dataset
    main(dataset_name)
