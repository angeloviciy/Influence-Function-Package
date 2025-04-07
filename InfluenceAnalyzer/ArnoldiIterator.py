import torch

class ArnoldiIterator:
    def __init__(self, hvp_func, r, tol=1e-8, device='cpu'):
        """
        Args:
            hvp_func: A function that takes a vector v and returns H*v.
            r: Number of Arnoldi iterations (defines the dimension of the Krylov subspace).
            tol: Tolerance for early termination.
            device: Torch device.
        """
        self.hvp_func = hvp_func
        self.r = r
        self.tol = tol
        self.device = device

    def run(self, v_init):
        """
        Performs the Arnoldi iteration.
        
        Args:
            v_init: The initial vector (flattened) from which to build the subspace.
        
        Returns:
            basis_mat: A matrix of shape (p, k) where each column is a basis vector.
            H_reduced: A (k x k) Hessenberg matrix approximating H in the subspace.
        """
        # Normalize the initial vector
        v_init = v_init / torch.norm(v_init)
        basis = [v_init]
        # Pre-allocate Hessenberg matrix (r+1 x r)
        H_matrix = torch.zeros(self.r + 1, self.r, device=self.device)
        
        for i in range(self.r):
            # Compute H*w for current basis vector
            w = self.hvp_func(basis[i])
            # Orthogonalize w against all previous basis vectors
            for j in range(i+1):
                H_matrix[j, i] = torch.dot(w, basis[j])
                w = w - H_matrix[j, i] * basis[j]
            norm_w = torch.norm(w)
            H_matrix[i+1, i] = norm_w
            if norm_w < self.tol:
                print("Arnoldi terminated early at iteration", i)
                break
            basis.append(w / norm_w)
        k = len(basis)
        # Stack basis vectors to form a matrix: shape (p, k)
        basis_mat = torch.stack(basis, dim=1)
        # Truncate H_matrix to the actual computed dimensions: k x k
        H_reduced = H_matrix[:k-1, :k-1]
        return basis_mat, H_reduced
