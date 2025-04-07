# Influence Function Analyzer

A minimal toolkit for computing influence functions in PyTorch models using Krylov subspaces and Conjugate Gradient.

## What’s in this directory

- **ArnoldiIterator.py**  
  Implements the Arnoldi iteration to build a Krylov subspace for Hessian approximation.

- **HessianVectorProduct.py**  
  Computes Hessian–vector products without forming the full Hessian.

- **HessianInverseCG.py**  
  Uses Conjugate Gradient to approximate applying the inverse Hessian to a vector and compute influence scores.

- **Projection.py**  
  Builds a projection matrix onto the top eigen-directions from the Arnoldi output.

- **InfluenceCalculator.py**  
  Computes influence scores by projecting gradients into the dominant subspace.

- **InfluenceFunctionAnalyzer.py**  
  High‑level class that ties everything together:  
  - Builds the Krylov subspace on a representative batch  
  - Computes per‑example gradients  
  - Computes influence scores (both via projection and via CG)  
  - Generates a pandas DataFrame report
  -can also be used to generate a CG-based influence calculation

## Requirements

- Python 3.7+
- look at requirements.txt

