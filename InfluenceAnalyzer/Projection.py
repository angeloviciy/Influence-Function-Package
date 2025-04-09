import torch

def compute_projection(basis_mat, H_reduced, p_tilde):
    """
    Args:
        basis_mat: Orthonormal basis matrix of shape (p, k) from Arnoldi iteration.
        H_reduced: Hessenberg matrix of shape (k, k) approximating H.
        p_tilde: Number of dominant eigenvalues/eigenvectors to retain.
    
    Returns:
        G: Projection matrix of shape (p_tilde, p) that maps a p-dimensional vector 
           to the dominant subspace.
        eigenvalues_top: The top p_tilde eigenvalues.
    """
    # Determine the effective number of iterations.
    k = basis_mat.shape[1]
    if k > 1:
        # Use only the first k-1 vectors to form a square matrix.
        H_sq = H_reduced[:k-1, :k-1]
        basis_sq = basis_mat[:, :k-1]
    else:
        H_sq = H_reduced
        basis_sq = basis_mat


    # Use eigen-decomposition; for a symmetric (or Hermitian) matrix, use eigh.
    eigenvalues, eigenvectors = torch.linalg.eigh(H_sq)
    
    # Sort eigenvalues in descending order of absolute value
    abs_eig = torch.abs(eigenvalues)
    sorted_indices = torch.argsort(abs_eig, descending=True)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Select the top p_tilde eigenvalues/eigenvectors
    eigenvalues_top = eigenvalues[:p_tilde]
    eigenvectors_top = eigenvectors[:, :p_tilde]  # shape: (k, p_tilde)
    
    # Construct projection matrix G:
    # Given basis_mat of shape (p, k), we define:
    #     G = (eigenvectors_top)^T * (basis_mat)^T,
    # so that for any vector g in R^p, the projected vector is:
    #     proj = G * g,  (of shape (p_tilde,)).
    G = torch.matmul(eigenvectors_top.T, basis_sq.T)
    return G, eigenvalues_top
