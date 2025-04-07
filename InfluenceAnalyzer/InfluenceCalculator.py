import torch

def compute_influence(G, eigenvalues_top, grad_x, grad_z):
    """
    Args:
        G: Projection matrix of shape (p_tilde, p).
        eigenvalues_top: Tensor of top eigenvalues of shape (p_tilde,).
        grad_x: Flattened gradient vector for training example x (shape: (p,)).
        grad_z: Flattened gradient vector for test/query example z (shape: (p,)).
    
    Returns:
        influence: The scalar influence score of training example x on example z.
    """
    # Project gradients into the dominant subspace
    proj_grad_x = torch.matmul(G, grad_x)  # shape: (p_tilde,)
    proj_grad_z = torch.matmul(G, grad_z)  # shape: (p_tilde,)


    damping = 1e-4  # or another small constant
    # Invert the eigenvalues (element-wise)
    inv_eigs = 1.0 / (eigenvalues_top + damping)
    # Compute the influence as the dot product with scaling by the inverse eigenvalues
    influence = torch.dot(proj_grad_z, proj_grad_x * inv_eigs)
    return influence
