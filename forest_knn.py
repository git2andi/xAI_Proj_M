import os
import numpy as np
import torch
import argparse
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA, KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.utils import resample
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import AdaBoostClassifier
from metric_learn import LMNN

root_path = "./database"

# Set the random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

results = {
    'method': [],
    'n_components': [],
    'k': [],
    'n_classifiers': [],
    'train_accuracies': [],
    'val_accuracies': [],
    'weights': [],
    'test_accuracy': [],
    'boosted_train_accuracy': [],
    'boosted_val_accuracy': []
}

# Load Data and prepare for further analysis
def load_embeddings_and_labels(dataset_name, root_path):
    """
    Load dataset embeddings and labels from the specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset.
        root_path (str): Root path where the dataset files are located.
    
    Returns:
        tuple: Training, validation, and test sets (X_train, y_train, X_val, y_val, X_test, y_test).
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
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_test, y_test = test_data['embeddings'], test_data['labels'].reshape(-1,)
    
    print(f"Loaded dataset {dataset_name} with shapes - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def select_important_features(X, y, n_features=100):
    """
    Select the most important features using mutual information or another feature selection method.
    
    Args:
        X (numpy.ndarray): Training data.
        y (numpy.ndarray): Training labels.
        n_features (int): Number of top features to select.
    
    Returns:
        numpy.ndarray: Reduced training data with selected features.
    """
    selector = SelectKBest(mutual_info_classif, k=n_features)
    X_new = selector.fit_transform(X, y)
    return X_new

def augment_features(X):
    """
    Create new features by combining existing ones (e.g., polynomial features).
    
    Args:
        X (numpy.ndarray): Training data.
    
    Returns:
        numpy.ndarray: Augmented training data with new features.
    """
    poly = PolynomialFeatures(degree=2, interaction_only=True)
    X_augmented = poly.fit_transform(X)
    return X_augmented

def create_bootstrap_samples(X, y, n_samples):
    """
    Create bootstrap samples from the training data.
    
    Args:
        X (numpy.ndarray): Training data.
        y (numpy.ndarray): Training labels.
        n_samples (int): Number of bootstrap samples to create.
    
    Returns:
        tuple: Bootstrap samples of data and labels.
    """
    print(f"Creating {n_samples} bootstrap samples...")
    X_samples, y_samples = [], []
    for i in range(n_samples):
        X_resampled, y_resampled = resample(X, y, random_state=None)  # Ensure randomness
        X_samples.append(X_resampled)
        y_samples.append(y_resampled)
        print(f"  Created bootstrap sample {i+1}/{n_samples}")
    return X_samples, y_samples


def apply_projections(X_samples, method='random', n_components=50, random_state=None):
    """
    Apply dimensionality reduction projections to the bootstrap samples.
    
    Args:
        X_samples (list): List of bootstrap samples.
        method (str): Projection method ('random', 'pca', or 'kernel_pca').
        n_components (int): Number of components for the projection.
        random_state (int or None): Random seed for reproducibility.
    
    Returns:
        tuple: Projected samples and the transformers used for projection.
    """
    print(f"Applying {method} projection with {n_components} components to bootstrap samples...")
    projected_samples = []
    transformers = []
    
    for i, X in enumerate(X_samples):
        if method == 'random':
            # Use a different random seed for each projection to ensure variability
            transformer = GaussianRandomProjection(n_components=n_components, random_state=random_state+i)
        elif method == 'pca':
            transformer = PCA(n_components=n_components)
        elif method == 'kernel_pca':
            transformer = KernelPCA(n_components=n_components, kernel='rbf')
        else:
            raise ValueError(f"Unknown method: {method}")
        
        X_projected = transformer.fit_transform(X)
        projected_samples.append(X_projected)
        transformers.append(transformer)
        
        print(f"  Applied {method} projection to sample {i+1}/{len(X_samples)}")
    
    return projected_samples, transformers

def apply_projections_to_test(X_test, transformers):
    """
    Apply the same projections to the test data.
    
    Args:
        X_test (numpy.ndarray): Test data.
        transformers (list): List of transformers used for projection.
    
    Returns:
        list: Projected test data samples.
    """
    print(f"Applying projections to test data...")
    X_test_projected_samples = []
    for i, transformer in enumerate(transformers):
        X_test_projected = transformer.transform(X_test)
        X_test_projected_samples.append(X_test_projected)
        print(f"  Applied projection to test data, projected shape: {X_test_projected.shape}")
    return X_test_projected_samples




def learn_distance_metric(X_train, y_train, max_iter):
    """
    Learn a distance metric that better captures relationships between data points.
    
    Args:
        X_train (numpy.ndarray): Training data.
        y_train (numpy.ndarray): Training labels.
        max_iter (int): Maximum number of iterations for LMNN.
    
    Returns:
        object: Learned distance metric model.
    """
    print(f"Starting distance metric learning with LMNN on data of shape: {X_train.shape}, with max_iter={max_iter}")
    
    # Adding verbose print statements to track progress
    class ProgressLMNN(LMNN):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.iteration = 0

        def _loss_grad(self, *args, **kwargs):
            self.iteration += 1
            print(f"Iteration {self.iteration}: Starting loss and gradient calculation...")
            result = super()._loss_grad(*args, **kwargs)
            print(f"Iteration {self.iteration}: Completed loss and gradient calculation.")
            return result

        def _fit_iter(self, *args, **kwargs):
            print(f"Iteration {self.iteration}: Starting LMNN iteration...")
            result = super()._fit_iter(*args, **kwargs)
            print(f"Iteration {self.iteration}: Completed LMNN iteration.")
            return result

    lmnn = ProgressLMNN(k=5, learn_rate=1e-6, max_iter=max_iter)
    lmnn.fit(X_train, y_train)
    print(f"Completed distance metric learning")
    return lmnn





#create KNN
class CustomKNN:
    """
    Custom k-Nearest Neighbors (kNN) classifier with GPU support.
    """
    def __init__(self, k=5, device='cuda'):
        self.k = k
        self.device = device

    def fit(self, X, y):
        """
        Fit the kNN model using the training data.
        
        Args:
            X (numpy.ndarray): Training data.
            y (numpy.ndarray): Training labels.
        """
        self.X_train = torch.tensor(X, device=self.device, dtype=torch.float)
        self.y_train = torch.tensor(y, device=self.device, dtype=torch.float)

    def predict(self, X):
        """
        Predict the labels for the input data using kNN.
        
        Args:
            X (numpy.ndarray): Input data for prediction.
        
        Returns:
            numpy.ndarray: Predicted labels.
        """
        X = torch.tensor(X, device=self.device, dtype=torch.float)
        distances = torch.cdist(X, self.X_train)
        neighbors = distances.argsort(dim=1)[:, :self.k]
        top_labels = self.y_train[neighbors]
        predictions = torch.mode(top_labels, dim=1).values
        return predictions.cpu().numpy()

def train_knn_models(X_samples, y_samples, X_val, y_val, k=5, device='cuda'):
    """
    Train multiple kNN models on the bootstrap samples and save training/validation accuracies.
    
    Args:
        X_samples (list): List of bootstrap samples of training data.
        y_samples (list): List of bootstrap samples of training labels.
        X_val (list): List of validation data projections.
        y_val (numpy.ndarray): Validation labels.
        k (int): Number of neighbors for kNN.
        device (str): Device to use for computation ('cuda' or 'cpu').
    
    Returns:
        list: Trained kNN models.
        list: Training accuracies of the kNN models.
        list: Validation accuracies of the kNN models.
    """
    print(f"Training {len(X_samples)} kNN models with k={k}...")
    classifiers = []
    train_accuracies = []
    val_accuracies = []
    
    for i, (X, y) in enumerate(zip(X_samples, y_samples)):
        knn = CustomKNN(k=k, device=device)
        knn.fit(X, y)
        classifiers.append(knn)
        
        # Calculate training accuracy
        y_train_pred = knn.predict(X)
        train_accuracy = accuracy_score(y, y_train_pred)
        train_accuracies.append(train_accuracy)
        
        # Calculate validation accuracy
        y_val_pred = knn.predict(X_val[i])
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_accuracies.append(val_accuracy)
        
        print(f"  Trained kNN model {i+1}/{len(X_samples)} - Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")
        print(f"    Validation predictions shape: {y_val_pred.shape}, Validation labels shape: {y_val.shape}")
    
    return classifiers, train_accuracies, val_accuracies




def train_boosted_knn(X_train, y_train, X_val, y_val, transformers, k=5, n_estimators=50):
    """
    Train a boosted kNN model using a custom boosting method and save its performance metrics.
    
    Args:
        X_train (numpy.ndarray): Training data.
        y_train (numpy.ndarray): Training labels.
        X_val (numpy.ndarray): Validation data.
        y_val (numpy.ndarray): Validation labels.
        transformers (list): List of transformers used for projection.
        k (int): Number of neighbors for kNN.
        n_estimators (int): Number of boosting rounds.
    
    Returns:
        list: Trained kNN models.
        list: Weights for each model.
    """
    boosted_knn_models = []
    boosted_weights = []

    for n in range(n_estimators):
        print(f"Training boosted kNN model {n+1}/{n_estimators}...")
        
        # Apply each transformer to the training data
        X_train_projected = [transformer.transform(X_train) for transformer in transformers]
        X_val_projected = [transformer.transform(X_val) for transformer in transformers]

        # Train kNN model on each projected data
        knn_models = []
        for X_proj in X_train_projected:
            knn = CustomKNN(k=k, device='cuda')
            knn.fit(X_proj, y_train)
            knn_models.append(knn)

        # Evaluate on validation data and calculate weights
        val_accuracies = []
        for knn, X_proj in zip(knn_models, X_val_projected):
            y_val_pred = knn.predict(X_proj)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_accuracies.append(val_accuracy)
        
        # Choose the best kNN model based on validation accuracy
        best_knn = knn_models[np.argmax(val_accuracies)]
        best_val_accuracy = max(val_accuracies)
        weight = best_val_accuracy / sum(val_accuracies)
        
        boosted_knn_models.append(best_knn)
        boosted_weights.append(weight)
        
        print(f"  Boosted kNN model {n+1}/{n_estimators} - Best Val Accuracy: {best_val_accuracy:.4f}, Weight: {weight:.4f}")
    
    return boosted_knn_models, boosted_weights




# use KNN to predict
def predict_ensemble(classifiers, X_test_samples):
    """
    Predict with an ensemble of kNN models and perform majority voting.
    
    Args:
        classifiers (list): List of trained kNN models.
        X_test_samples (list): List of projected test data samples.
    
    Returns:
        numpy.ndarray: Final predicted labels after majority voting.
    """
    print(f"Predicting with ensemble of {len(classifiers)} kNN models...")
    predictions = np.zeros((X_test_samples[0].shape[0], len(classifiers)))
    for i, (clf, X_test) in enumerate(zip(classifiers, X_test_samples)):
        predictions[:, i] = clf.predict(X_test)
        print(f"  Predicted with kNN model {i+1}/{len(classifiers)}")
    final_predictions = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=predictions)
    return final_predictions



def predict_weighted_ensemble(models, X_samples, weights):
    """
    Predict with an ensemble of kNN models using weighted voting.
    
    Args:
        models (list): List of trained kNN models.
        X_samples (list): Projected test data samples.
        weights (list): List of weights for each classifier.
    
    Returns:
        numpy.ndarray: Final predicted labels after weighted voting.
    """
    print(f"Predicting with weighted ensemble of {len(models)} kNN models...")
    predictions = np.zeros((X_samples[0].shape[0], len(models)))
    for i, (model, X) in enumerate(zip(models, X_samples)):
        predictions[:, i] = model.predict(X)
        print(f"  Predicted with kNN model {i+1}/{len(models)}")
    
    weighted_votes = np.zeros(predictions.shape[0])
    for i, weight in enumerate(weights):
        weighted_votes += weight * predictions[:, i]
    final_predictions = np.round(weighted_votes / sum(weights)).astype(int)
    return final_predictions






# Create csv and save results
def evaluate_and_save_results(classifiers, X_test_samples, y_test, method, n_components, k, output_file, weights=None):
    """
    Evaluate the ensemble of kNN models and save the results.
    
    Args:
        classifiers (list): List of trained kNN models.
        X_test_samples (list): List of projected test data samples.
        y_test (numpy.ndarray): True labels for the test data.
        method (str): Projection method used.
        n_components (int): Number of components for the projection.
        k (int): Number of neighbors for kNN.
        output_file (str): Path to the output file for saving results.
        weights (list, optional): List of weights for weighted voting. Defaults to None.
    """
    if weights:
        y_pred = predict_weighted_ensemble(classifiers, X_test_samples, weights)
    else:
        y_pred = predict_ensemble(classifiers, X_test_samples)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy for method={method}, n_components={n_components}, k={k}: {accuracy:.4f}")
    
    result = {
        'method': method,
        'n_components': n_components,
        'k': k,
        'accuracy': accuracy
    }
    df = pd.DataFrame([result])
    df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
    print(f"Results saved to {output_file}")



# main
def main_forest_knn(dataset_name):
    """
    Main function to run Forest-kNN on the specified dataset and save results.
    
    Args:
        dataset_name (str): Name of the dataset to use.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = load_embeddings_and_labels(dataset_name, root_path)
    
    print(f"Initial training data shape: {X_train.shape}")
    
    n_classifiers_values = [20] #, 30]  # Different numbers of classifiers to experiment with
    n_components_list = [50] #, 100]  # More varied component counts to test different dimensionality reductions
    k_values = [5] #, 10]  # Different k values to experiment with
    methods = ['random', 'pca', 'kernel_pca']
    output_file = f'{dataset_name}_forest_knn_results.csv'
    
    results = {
        'method': [],
        'n_components': [],
        'k': [],
        'n_classifiers': [],
        'train_accuracies': [],
        'val_accuracies': [],
        'weights': [],
        'test_accuracy': [],
        'boosted_train_accuracy': [],
        'boosted_val_accuracy': []
    }
    
    for n_classifiers in n_classifiers_values:
        X_train_selected = select_important_features(X_train, y_train)
        print(f"After feature selection, training data shape: {X_train_selected.shape}")
        
        X_train_augmented = augment_features(X_train_selected)
        print(f"After feature augmentation, training data shape: {X_train_augmented.shape}")
        
        X_samples, y_samples = create_bootstrap_samples(X_train_augmented, y_train, n_classifiers)
        
        # Apply learned distance metric to the augmented training data
        print("Starting distance metric learning...")
        distance_metric_model = learn_distance_metric(X_train_augmented, y_train, max_iter=2)
        X_train_transformed = distance_metric_model.transform(X_train_augmented)
        print(f"Completed distance metric learning, transformed training data shape: {X_train_transformed.shape}")
        
        # Transform validation and test data using the same distance metric model
        X_val_selected = select_important_features(X_val, y_val)
        X_val_augmented = augment_features(X_val_selected)
        X_val_transformed = distance_metric_model.transform(X_val_augmented)
        print(f"Transformed validation data shape: {X_val_transformed.shape}")
        
        X_test_selected = select_important_features(X_test, y_test)
        X_test_augmented = augment_features(X_test_selected)
        X_test_transformed = distance_metric_model.transform(X_test_augmented)
        print(f"Transformed test data shape: {X_test_transformed.shape}")
        
        for method in methods:
            for n_components in n_components_list:
                for k in k_values:
                    print(f"\nStarting evaluation for method={method}, n_components={n_components}, k={k}, n_classifiers={n_classifiers}...")
                    X_projected_samples, transformers = apply_projections(X_samples, method=method, n_components=n_components, random_state=42)
                    X_val_projected = apply_projections_to_test(X_val_transformed, transformers)
                    X_test_projected = apply_projections_to_test(X_test_transformed, transformers)
                    X_train_projected = [transformer.transform(X_train_transformed) for transformer in transformers]


                    classifiers, train_accuracies, val_accuracies = train_knn_models(X_projected_samples, y_samples, X_val_projected, y_val, k=k, device='cuda')
            
                    # Standard ensemble
                    evaluate_and_save_results(classifiers, X_test_projected, y_test, method, n_components, k, output_file)
            
                    # Save results for standard ensemble
                    results['method'].append(method)
                    results['n_components'].append(n_components)
                    results['k'].append(k)
                    results['n_classifiers'].append(n_classifiers)
                    results['train_accuracies'].append(train_accuracies)
                    results['val_accuracies'].append(val_accuracies)
            
                    # Train custom boosted kNN models
                    boosted_knn_models, boosted_weights = train_boosted_knn(X_train_transformed, y_train, X_val_transformed, y_val, transformers, k=k)
                    boosted_train_accuracy = accuracy_score(y_train, predict_weighted_ensemble(boosted_knn_models, X_train_projected, boosted_weights))
                    boosted_val_accuracy = accuracy_score(y_val, predict_weighted_ensemble(boosted_knn_models, X_val_projected, boosted_weights))

                    results['boosted_train_accuracy'].append(boosted_train_accuracy)
                    results['boosted_val_accuracy'].append(boosted_val_accuracy)
            
                    # Evaluate boosted ensemble
                    boosted_test_accuracy = accuracy_score(y_test, predict_weighted_ensemble(boosted_knn_models, X_test_projected, boosted_weights))
                    evaluate_and_save_results(boosted_knn_models, X_test_projected, y_test, method, n_components, k, output_file, weights=boosted_weights)
                    results['test_accuracy'].append(boosted_test_accuracy)
    
    # Save all results to a CSV file for further analysis
    df = pd.DataFrame(results)
    print(f"Results dataframe: {df}")
    df.to_csv(output_file, mode='w', index=False)
    print(f"All results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forest-kNN Classification with PyTorch")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100", "dermamnist", "breastmnist"], help="Dataset to use")
    args = parser.parse_args()

    dataset_name = args.dataset
    print(f"Starting Forest-kNN on dataset {dataset_name}")
    main_forest_knn(dataset_name)
    print("Forest-kNN evaluation completed.")
