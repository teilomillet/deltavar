# deltavar
https://arxiv.org/pdf/2502.14698

# Installation
```bash
uv add deltavar
```

## Overview
This library provides tools to calculate the Delta Variance, an approximation for the change in a model's prediction variance due to uncertainty in its parameters, often linked to epistemic uncertainty. It leverages `torch.func` for functional gradient computations.

## Core Functionality (`deltavar/core.py`)
The main entry point for calculating the Delta Variance.

*   `get_qoi_gradient(qoi_functional, params, model, z)`: Computes the gradient of a user-defined Quantity of Interest (QoI) functional with respect to the model parameters (`params`) for a specific input `z`.
*   `calculate_delta_variance(qoi_gradient_pytree, sigma_strategy, sigma_options, ...)`: Calculates the Delta Variance \( \\Delta Var = \\nabla q^T \\Sigma \\nabla q \). It takes the QoI gradients and a strategy for defining the Sigma matrix (parameter covariance approximation).

## Sigma Strategies (`deltavar/sigma.py`)
This module handles the definition and calculation of the Sigma matrix used in the Delta Variance computation.

**Sigma Application:**
Functions to compute \( \\nabla q^T \\Sigma \\nabla q \) for different Sigma structures:
*   `apply_identity_sigma(flat_grads)`: Assumes \( \\Sigma = I \). Returns \( ||\\nabla q||^2 \).
*   `apply_diagonal_sigma(flat_grads, diag_elements)`: Assumes \( \\Sigma \) is diagonal.
*   `apply_full_sigma(flat_grads, sigma_matrix)`: Uses a precomputed dense \( \\Sigma \) matrix.

**Sigma Calculation (using Empirical FIM):**
Functions to calculate the Empirical Fisher Information Matrix (FIM) and its inverse, which can serve as \( \\Sigma \). These utilize `torch.func.vmap` for efficiency.
*   `calculate_empirical_fim_diagonal(...)`: Computes the diagonal elements of the FIM: \( F_{ii} = E_D [ (\\nabla_i \\log p(D|\\theta))^2 ] \).
*   `calculate_inverse_fim_diagonal(..., damping)`: Computes \( \\text{diag}( (F_{diag} + \\lambda I)^{-1} ) \).
*   `calculate_empirical_fim_full(...)`: Computes the full FIM: \( F = E_D [ \\nabla \\log p(D|\\theta) \\nabla \\log p(D|\\theta)^T ] \). Computationally demanding (O(P^2) memory).
*   `calculate_inverse_fim_full(..., damping)`: Computes \( (F + \\lambda I)^{-1} \). Computationally demanding (O(P^3) for inversion).

**Sigma Specification in `calculate_delta_variance`:**
The `sigma_strategy` argument in `calculate_delta_variance` accepts:
*   `'identity'`: Uses `apply_identity_sigma`.
*   `'diagonal_empirical_fim'`: Calculates and uses the inverse diagonal FIM via `calculate_inverse_fim_diagonal`. Requires likelihood function, model, params, dataset.
*   `'inverse_full_empirical_fim'`: Calculates and uses the inverse full FIM via `calculate_inverse_fim_full`. Requires likelihood function, model, params, dataset.
*   `torch.Tensor` (1D): Interpreted as diagonal elements for `apply_diagonal_sigma`.
*   `torch.Tensor` (2D): Interpreted as the full matrix for `apply_full_sigma`.
*   `Callable`: A custom function `f(flat_grads) -> scalar` implementing the \( \\nabla q^T \\Sigma \\nabla q \) calculation.

## Testing (`tests/`)
Unit tests are provided to ensure the correctness of the core components.
*   `test_core.py`: Contains tests for `get_qoi_gradient` and `calculate_delta_variance` with various Sigma strategies.
*   `test_sigma.py`: Contains tests for Sigma application functions and FIM calculation routines.

## Usage Example

Here's a basic example demonstrating how to calculate the Delta Variance, illustrating the steps involved in estimating epistemic uncertainty using this method.

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from deltavar import get_qoi_gradient, calculate_delta_variance

# 1. Define Model, Functionals, and Data

# We start with a standard PyTorch model.
model = nn.Linear(10, 1)

# Define the Quantity of Interest (QoI). This is the specific scalar value
# derived from the model's output whose uncertainty we want to estimate.
# Here, it's simply the model's output for a specific input `z`.
# The functional must accept `params`, `model`, and the input `z`.
# It uses `torch.func.functional_call` to run the model statelessly.
def qoi_functional(params, model, z):
    return torch.func.functional_call(model, params, (z,))

# Define the Log Likelihood functional. This is only needed if you plan to
# calculate Sigma using the Fisher Information Matrix (FIM).
# It computes the log-likelihood of a single data point given the model parameters.
# Like the QoI, it uses `functional_call`.
def log_likelihood_functional(params, model, data_point):
    x, y = data_point
    prediction = torch.func.functional_call(model, params, (x,))
    # Example: Gaussian log-likelihood (proportional to negative MSE)
    neg_mse = -0.5 * torch.mean((prediction - y)**2)
    return neg_mse

# Prepare stateless parameters. `torch.func` requires model parameters
# as a separate argument (typically a dictionary or other PyTree).
# We detach and clone them from the stateful model.
params = {k: v.detach().clone() for k, v in model.named_parameters()}

# Define the specific input `z` for which we want the QoI's uncertainty.
z = torch.randn(1, 10)

# Prepare a dataset. This is needed only for FIM-based Sigma calculations.
# It should be representative of the data used to train the model.
# A DataLoader is convenient for batching during FIM calculation.
data_x = torch.randn(100, 10)
data_y = model(data_x).detach() + torch.randn(100, 1) * 0.1 # Simulate some data
dataset = TensorDataset(data_x, data_y)
data_loader = DataLoader(dataset, batch_size=32)

# 2. Calculate QoI Gradient (∇q)

# Compute the gradient of the QoI with respect to the model parameters.
# This gradient vector (∇q) measures how sensitive the QoI is to changes
# in each parameter. It's the first key component of the Delta Variance formula.
qoi_grads_tree = get_qoi_gradient(qoi_functional, params, model, z)
# The result matches the structure of `params` (e.g., dict: {'weight': ..., 'bias': ...})

# 3. Calculate Delta Variance (∇q^T Σ ∇q)

# Now, combine the QoI gradient (∇q) with a chosen Sigma matrix (Σ)
# to estimate the variance. Σ approximates the covariance of the parameters.

# Option A: Identity Sigma (Simplest Case)
# Assumes Σ = I. The Delta Variance becomes the squared L2 norm of the gradient.
# This is computationally cheap but ignores parameter correlations and scaling.
delta_var_identity = calculate_delta_variance(
    qoi_gradient_pytree=qoi_grads_tree,
    sigma_strategy='identity'
)
print(f"Delta Variance (Identity Sigma): {delta_var_identity.item():.4f}")

# Option B: Inverse Diagonal Empirical FIM Sigma (Common Approximation)
# Calculates Σ ≈ diag( (diag(FIM) + λI)^-1 ).
# The FIM captures the curvature of the log-likelihood, related to parameter uncertainty.
# Using its inverse diagonal is a common, efficient approximation for Σ.
# Requires the log-likelihood functional and the dataset used for training.
delta_var_diag_fim = calculate_delta_variance(
    qoi_gradient_pytree=qoi_grads_tree,
    sigma_strategy='diagonal_empirical_fim',
    sigma_options={'damping': 1e-5, 'vmap_batch_size': 64}, # Options for FIM calc
    log_likelihood_functional=log_likelihood_functional,
    params=params,
    model=model,
    dataset=data_loader, # Pass the dataset iterable
)
print(f"Delta Variance (Diagonal FIM Sigma): {delta_var_diag_fim.item():.4f}")

# Option C: Providing a Pre-calculated Sigma
# If you have a pre-computed Σ (e.g., from another method, or a specific
# structure you want to test), you can pass it directly as a tensor.
# 1D tensor: Assumes Σ is diagonal.
# 2D tensor: Assumes Σ is a full matrix.
num_params = sum(p.numel() for p in params.values())
my_diag_sigma = torch.ones(num_params) * 0.1 # Example: Simple diagonal Σ
delta_var_custom_diag = calculate_delta_variance(
    qoi_gradient_pytree=qoi_grads_tree,
    sigma_strategy=my_diag_sigma
)
print(f"Delta Variance (Custom Diagonal Sigma): {delta_var_custom_diag.item():.4f}")
```
