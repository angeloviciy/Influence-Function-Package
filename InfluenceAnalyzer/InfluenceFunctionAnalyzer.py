import torch

from InfluenceAnalyzer.HessianVectorProduct import hessian_vector_product
from InfluenceAnalyzer.ArnoldiIterator import ArnoldiIterator
from InfluenceAnalyzer.Projection import compute_projection
from InfluenceAnalyzer.InfluenceCalculator import compute_influence
from InfluenceAnalyzer.HessianInverseCG import compute_influence_cg
import pandas as pd


class InfluenceFunctionAnalyzer:
    def __init__(self, model, loss_fn, params, r, p_tilde, device='cpu'):
        """
        Args:
            model: A PyTorch model.
            loss_fn: Loss function.
            params: Iterable of model parameters.
            r: Number of Arnoldi iterations (size of Krylov subspace).
            p_tilde: Number of dominant eigen-directions to retain.
            device: Torch device.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.params = list(params)
        self.r = r
        self.p_tilde = p_tilde
        self.device = device
        self.G = None
        self.eigenvalues_top = None

    def hvp(self, loss, v):
        return hessian_vector_product(loss, self.model, self.params, v)

    def build_krylov_subspace(self, input_batch, target_batch):
        """
        Build and cache the projection matrix G and eigenvalues using a representative data batch.
        
        Args:
            input_batch: Inputs (batch) for computing the loss.
            target_batch: Corresponding targets.
        """
        # Compute loss on the provided batch
        outputs = self.model(input_batch)
        loss = self.loss_fn(outputs, target_batch)
        
        # Determine the total number of parameters (flattened)
        total_params = sum(p.numel() for p in self.params)

        
        # Initialize a random vector v_init of the correct size
        v_init = torch.randn(total_params, device=self.device)
        
        # Define a function that computes H*v using the current loss
        def hvp_func(v):
            return self.hvp(loss, v)
        
        # Run the Arnoldi iteration
        arnoldi_iter = ArnoldiIterator(hvp_func, self.r, device=self.device)
        basis_mat, H_reduced = arnoldi_iter.run(v_init)
        
        # Compute the projection matrix G and select top p_tilde eigenvalues
        self.G, self.eigenvalues_top = compute_projection(basis_mat, H_reduced, self.p_tilde)

    def _compute_flat_grad(self, input_data, target):
        """
        Computes the flattened gradient vector for a single example (or batch of size 1).
        """
        self.model.zero_grad()
        output = self.model(input_data)
        loss = self.loss_fn(output, target)
        grads = torch.autograd.grad(loss, self.params, retain_graph=True)
        flat_grad = torch.cat([g.reshape(-1) for g in grads])
        return flat_grad

    def compute_influence_for_examples(self, train_example, test_example):
        """
        Computes the influence score for a training example on a test example.
        
        Args:
            train_example: Tuple (input, target) for a training example.
            test_example: Tuple (input, target) for a test or query example.
        
        Returns:
            influence: Scalar influence score.
        """
        train_input, train_target = train_example
        test_input, test_target = test_example
        
        grad_train = self._compute_flat_grad(train_input, train_target)
        grad_test = self._compute_flat_grad(test_input, test_target)
        
        influence = compute_influence(self.G, self.eigenvalues_top, grad_train, grad_test)
        return influence

    def generate_influence_report(self, train_dataset, test_example, indices=None):
        """
        Computes influence scores for all training examples with respect to a given test example
        and returns a sorted report.
        
        Args:
            train_dataset: Iterable dataset of training examples, each as (input, target).
            test_example: Tuple (input, target) for the test example.
            indices: (Optional) Identifiers for the training examples.
        
        Returns:
            report_df: A pandas DataFrame sorted by influence score.
        """
        influences = []
        ids = []
        for idx, train_example in enumerate(train_dataset):
            inf = self.compute_influence_for_examples(train_example, test_example)
            influences.append(inf.item())
            ids.append(idx if indices is None else indices[idx])
        df = pd.DataFrame({
            'Training Example': ids,
            'Influence Score': influences
        })
        return df.sort_values(by='Influence Score', ascending=False)
    
    def compute_influence_cg(self, train_example, test_example):
        """
        Computes the influence score for a training example on a test example using CG.
        """
        train_input, train_target = train_example
        test_input, test_target = test_example

        grad_train = self._compute_flat_grad(train_input, train_target)
        grad_test = self._compute_flat_grad(test_input, test_target)

        influence = compute_influence_cg(self.loss_fn, self.model, self.params, grad_train, grad_test)
        return influence
    
