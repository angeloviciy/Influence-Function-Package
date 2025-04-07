import torch

def hessian_vector_product(loss, model, params, v):
    """
    Compute the Hessian-vector product H*v without materializing H.
    
    Args:
        loss: Scalar loss computed for a data batch.
        model: The PyTorch model.
        params: Iterable of model parameters.
        v: A vector (flattened tensor) to be multiplied by the Hessian.
    
    Returns:
        hv_vector: The product H*v as a flattened tensor.
    """
    # First, compute the gradient (create_graph=True for higher-order derivatives)
    grad_params = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    grad_vector = torch.cat([g.reshape(-1) for g in grad_params])
    
    # Compute the inner product <grad, v>
    dot = torch.dot(grad_vector, v)
    
    # Compute gradient of the dot product (this is the Hessian-vector product)
    hv = torch.autograd.grad(dot, params, retain_graph=True)
    hv_vector = torch.cat([h.reshape(-1) for h in hv])
    return hv_vector
