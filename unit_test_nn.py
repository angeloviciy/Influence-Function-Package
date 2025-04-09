import unittest
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
import torchvision.transforms as transforms
from scipy.stats import spearmanr, kendalltau
import numpy as np
import itertools


# Import the analyzer and related functions from your InfluenceAnalyzer package.
from InfluenceAnalyzer.InfluenceFunctionAnalyzer import InfluenceFunctionAnalyzer

#############################
# Define two MNIST models with varying depth.
#############################

class SimpleNet(nn.Module):
    # A one-layer MLP (flatten and linear)
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(28*28, 10)
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

class DeepNet(nn.Module):
    # A two-layer MLP with one hidden layer.
    def __init__(self):
        super(DeepNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

#############################
# Helper functions for training and poisoning.
#############################

def train_model(model, dataset, num_epochs=5, lr=0.01, batch_size=64):
    #train WITH l_2 regularization
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()
            #l2 regularization
            l2_reg = sum(p.norm()**2 for p in model.parameters())
            loss += l2_reg * 0.001
            loss.backward()
            optimizer.step()

    return model

def poison_dataset(dataset, poison_fraction=0.2):
    """
    Randomly flip the label of a fraction of the dataset examples.
    Returns a new dataset (TensorDataset) and the list of poisoned indices.
    Increasing the fraction (e.g. 0.2) makes the poison effect stronger.
    """
    # Unpack the dataset into tensors.
    data = dataset.data.clone()  # MNIST data: [N, 28, 28]
    targets = dataset.targets.clone()
    N = len(targets)
    num_poison = int(poison_fraction * N)
    poisoned_indices = random.sample(range(N), num_poison)
    
    # For each chosen index, set the label to a different random value.
    for idx in poisoned_indices:
        orig = int(targets[idx])
        possible = list(range(10))
        possible.remove(orig)
        targets[idx] = torch.tensor(random.choice(possible))
    
    # Convert images to float and normalize using ToTensor.
    data_tensor = torch.stack([img.float()/255.0 for img in data])
    new_dataset = TensorDataset(data_tensor, targets)
    return new_dataset, poisoned_indices

def compute_precision_at_k(ranked_indices, true_poisoned_set, k):
    """Compute precision at top k given a ranking of indices (descending order)."""
    top_k = ranked_indices[:k]
    hits = sum(1 for idx in top_k if idx in true_poisoned_set)
    return hits / k

#############################
# Global Device Setting
#############################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################
# The Test Suite: MNIST Influence with Hyperparameter Tuning and Detailed Stats
#############################

class TestMNISTInfluenceHyperparam(unittest.TestCase):
    def setUp(self):
        # Set random seeds for reproducibility.
        torch.manual_seed(42)
        random.seed(42)
        
        # Load MNIST training and test sets.
        transform = transforms.Compose([transforms.ToTensor()])
        full_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        # Use a smaller subset for speed.
        self.train_subset = Subset(full_train, list(range(2000)))
        self.test_subset = Subset(test_set, list(range(500)))
        
        # Poison the training subset with a higher poison fraction.
        # Stronger poison: 20% of training examples have flipped labels.
        poisoned_full, self.poisoned_indices = poison_dataset(self.train_subset.dataset, poison_fraction=0.2)
        # Restrict to the same indices used in self.train_subset.
        self.poisoned_train = Subset(poisoned_full, list(range(2000)))
        
        # For influence evaluation, pick one test example.
        self.test_example = self.test_subset[0]  # (image, label)
        self.test_input = self.test_example[0].unsqueeze(0).to(device)
        self.test_target = torch.tensor([self.test_example[1]]).to(device)
        
        # Build a representative batch from the training data (for Krylov subspace).
        rep_indices = list(range(64))
        # The dataset has data in its .data and .targets.
        self.rep_batch_inputs = torch.stack([self.train_subset.dataset.data[i] for i in rep_indices]).float() / 255.0
        # Add a channel dimension if necessary.
        if self.rep_batch_inputs.dim() == 3:
            self.rep_batch_inputs = self.rep_batch_inputs.unsqueeze(1)
        self.rep_batch_targets = torch.tensor([self.train_subset.dataset.targets[i] for i in rep_indices])
        
        # For influence scoring, we will evaluate on 500 examples (subset).
        self.influence_eval_indices = list(range(500))
        
       # Define the grid values for each hyperparameter.
        r_values = [50, 100, 150, 200]
        p_tilde_values = [10, 20, 100, 200]
        tol_values = [1e-8, 1e-7]
        damping_values = [1e-3, 1e-2]  # Adding an extra candidate for damping.

        # Use itertools.product to generate all possible combinations.
        self.arnoldi_params = [
            {'r': r, 'p_tilde': p, 'tol': tol, 'damping': damp}
            for r, p, tol, damp in itertools.product(r_values, p_tilde_values, tol_values, damping_values)
        ]

    def _evaluate_influence(self, model, analyzer, dataset):
        """
        Compute self-influence scores for each example in dataset.
        Returns a list of (index, influence score).
        """
        model.eval()
        scores = []
        for idx in range(len(dataset)):
            inp, target = dataset[idx]
            inp = inp.unsqueeze(0).to(device)
            target = torch.tensor([target]).to(device)
            score = analyzer.compute_influence_for_examples((inp, target), (inp, target))
            scores.append((idx, score.item()))
        return scores

    def _analyze_scores(self, scores, poisoned_set):
        """
        Given scores (list of (index, score)), compute:
        - Precision at top-50,
        - Spearman and Kendall rank correlation between binary poison indicator and absolute influence,
        - Mean and standard deviation of influence scores for poisoned and clean examples.
        Returns a dict of metrics.
        """
        scores_sorted = sorted(scores, key=lambda x: abs(x[1]), reverse=True)
        ranked_indices = [idx for idx, _ in scores_sorted]
        precision50 = compute_precision_at_k(ranked_indices, poisoned_set, k=50)
        
        binary_indicator = [1 if idx in poisoned_set else 0 for idx, _ in scores_sorted]
        influence_values = [abs(score) for _, score in scores_sorted]
        rho, _ = spearmanr(binary_indicator, influence_values)
        tau, _ = kendalltau(binary_indicator, influence_values)
        
        # Separate scores for poisoned and clean.
        poisoned_scores = [score for idx, score in scores if idx in poisoned_set]
        clean_scores = [score for idx, score in scores if idx not in poisoned_set]
        metrics = {
            'precision_at_50': precision50,
            'spearman_rho': rho,
            'kendall_tau': tau,
            'poisoned_mean': np.mean(poisoned_scores) if poisoned_scores else None,
            'poisoned_std': np.std(poisoned_scores) if poisoned_scores else None,
            'clean_mean': np.mean(clean_scores) if clean_scores else None,
            'clean_std': np.std(clean_scores) if clean_scores else None,
        }
        return metrics

    def _run_single_hyperparam_test(self, model_class, hyperparams):
        """
        Train a model (using the given model_class) on the poisoned training subset,
        build the InfluenceFunctionAnalyzer using the provided hyperparameters,
        and evaluate influence scores on a subset of training examples.
        Returns the computed metrics.
        """
        print("\n--- Running test with hyperparams:", hyperparams, "---")
        model = model_class().to(device)
        model = train_model(model, self.poisoned_train, num_epochs=30, lr=0.01, batch_size=64)
        # Set up the analyzer with the current Arnoldi hyperparameters.
        analyzer = InfluenceFunctionAnalyzer(model, nn.CrossEntropyLoss(), model.parameters(),
                                             r=hyperparams['r'], p_tilde=hyperparams['p_tilde'], damping=hyperparams['damping'], device=device)
        rep_inputs = self.rep_batch_inputs.to(device)
        rep_targets = self.rep_batch_targets.to(device)
        analyzer.build_krylov_subspace(rep_inputs, rep_targets)
        
        # Evaluate influence on a subset of 500 training samples.
        subset_for_influence = Subset(self.poisoned_train, self.influence_eval_indices)
        scores = self._evaluate_influence(model, analyzer, subset_for_influence)
        # Determine poisoned examples within these indices.
        poisoned_in_subset = set(i for i in self.poisoned_indices if i < 500)
        metrics = self._analyze_scores(scores, poisoned_in_subset)
        print("Metrics:", metrics)
        return metrics

    def test_hyperparameter_tuning(self):
        """
        Run the full influence computation pipeline on a trained MNIST model (using DeepNet)
        for several hyperparameter configurations.
        Log detailed statistics including precision at top 50, rank correlations, and statistics on the influence scores.
        """
        results = {}
        # Run tests for each hyperparameter setting.
        for params in self.arnoldi_params:
            metrics = self._run_single_hyperparam_test(DeepNet, params)
            key = f"r={params['r']}_ptilde={params['p_tilde']}_tol={params['tol']}"
            results[key] = metrics
        
        # Optionally, assert that at least one configuration achieves a minimally acceptable precision
        acceptable = False
        for key, m in results.items():
            if m['precision_at_50'] >= 0.3 and m['spearman_rho'] is not None and m['spearman_rho'] >= 0.1:
                acceptable = True
                print(f"Configuration {key} meets minimal acceptance thresholds.")
        self.assertTrue(acceptable, "None of the hyperparameter configurations reached acceptable influence retrieval performance.")
        
        # Log the full results for further analysis.
        print("Hyperparameter Tuning Results:")
        for key, metrics in results.items():
            print(f"{key}: {metrics}")

if __name__ == "__main__":
    unittest.main()
