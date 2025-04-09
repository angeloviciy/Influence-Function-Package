import torch
from InfluenceAnalyzer.HessianVectorProduct import hessian_vector_product


def hvp_cg_solver(hvp_fn, b, cg_iters=50, residual_tol=1e-7):
    """
    Solve H x = b for x using Conjugate Gradient, where H is represented by hvp_fn.

    Args:
        hvp_fn: function that takes a vector v and returns H v.
        b: right-hand side vector (e.g., gradient vector) of shape (p,).
        cg_iters: maximum number of CG iterations.
        residual_tol: tolerance for the residual norm.

    Returns:
        x: approximate solution to H x = b, i.e., H^{-1} b.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rsold = torch.dot(r, r)

    for i in range(cg_iters):
        Hp = hvp_fn(p)
        alpha = rsold / (torch.dot(p, Hp) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Hp
        rsnew = torch.dot(r, r)
        if torch.sqrt(rsnew) < residual_tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x


def compute_influence_cg(loss, model, params, grad_x, grad_z, cg_iters=100, tol=1e-6, damping=1e-3):
    """
    Compute the influence of a training example x on a test example z using CG to approximate H^{-1}.

    Influence ≈ - grad_z^T H^{-1} grad_x

    Args:
        loss: scalar loss (e.g., on the full training data or a batch).
        model: the PyTorch model.
        params: list of model parameters.
        grad_x: flattened gradient vector for training example x (shape: (p,)).
        grad_z: flattened gradient vector for test example z (shape: (p,)).
        cg_iters: maximum CG iterations.
        tol: residual tolerance for CG.

    Returns:
        influence: scalar influence score.
    """
    # Define Hessian-vector product function for the full loss
    def hvp_fn(v):
        return hessian_vector_product(loss, model, params, v) + damping * v


    # Solve H^{-1} grad_x via CG
    Hinv_grad_x = hvp_cg_solver(hvp_fn, grad_x, cg_iters=cg_iters, residual_tol=tol)

    # Influence formula (with negative sign)
    influence = -torch.dot(grad_z, Hinv_grad_x)
    return influence
