import torch
from torch.func import grad, functional_call
from typing import Callable, Dict, Any, Union, Optional, Iterable, List
import logging
from .utils import flatten_pytree, TensorTree
from .sigma import (
    apply_identity_sigma,
    apply_diagonal_sigma,
    apply_full_sigma,
    calculate_inverse_fim_diagonal,
    calculate_inverse_fim_full, # Import the full FIM calculator
)

logger = logging.getLogger(__name__)

# Type alias for the quantity of interest function
# Takes params, model, input_data -> scalar tensor
QoiFunctional = Callable[[Dict[str, torch.Tensor], torch.nn.Module, Any], torch.Tensor]

# Type alias for the log likelihood function
# Takes params, model, data_point -> scalar tensor
LogLikelihoodFunctional = Callable[[Dict[str, torch.Tensor], torch.nn.Module, Any], torch.Tensor]


def get_qoi_gradient(
    qoi_functional: QoiFunctional,
    params: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    z: Any, # Single data point input for the QoI
) -> Dict[str, torch.Tensor]:
    """
    Computes the gradient of the Quantity of Interest (QoI) w.r.t. model parameters.

    Uses `torch.func.grad` for functional gradient computation.

    Args:
        qoi_functional: A callable `f(params, model, z)` that takes model parameters (PyTree),
                        the model instance, and a single input data point `z`.
                        It must return a scalar tensor representing the QoI for `z`.
        params: A dictionary (or PyTree) of the model\'s parameters.
                Example: `{k: v.detach().clone() for k, v in model.named_parameters()}`.
                These tensors should *not* require gradients themselves; `torch.func.grad` handles this.
        model: The PyTorch model (`nn.Module`). It is used statelessly via `functional_call`
               inside the `qoi_functional`.
        z: The specific input data point (e.g., a tensor, tuple of tensors) for which to
           calculate the QoI gradient. Its structure must be compatible with `qoi_functional`.

    Returns:
        A dictionary (or PyTree matching `params`) containing the gradients w.r.t.
        each parameter.
    """
    # Define the function whose gradient we need w.r.t. the *first* argument (params)
    def func_to_grad(p: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Pass the static model and the specific input z
        return qoi_functional(p, model, z)

    # Compute the gradient using torch.func.grad
    # The gradient will have the same structure as `params`
    param_grads = grad(func_to_grad)(params)

    return param_grads


def calculate_delta_variance(
    qoi_gradient_pytree: TensorTree,
    sigma_strategy: Union[str, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]] = 'identity',
    sigma_options: Optional[Dict[str, Any]] = None,
    # Optional args needed for *calculating* sigma (if not precomputed)
    log_likelihood_functional: Optional[LogLikelihoodFunctional] = None,
    params: Optional[Dict[str, torch.Tensor]] = None,
    model: Optional[torch.nn.Module] = None,
    dataset: Optional[Iterable[Any]] = None,
    device: Optional[torch.device] = None, # Added device hint
) -> torch.Tensor:
    """
    Calculates the Delta Variance based on QoI gradients and a Sigma strategy.

    The Delta Variance approximates the epistemic variance of the QoI and is computed as:
    Delta Variance = grad_qoi^T * Sigma * grad_qoi

    Args:
        qoi_gradient_pytree: The gradients of the QoI w.r.t. parameters.
                             Can be a tensor or a PyTree (dict/list/tuple) of tensors.
                             Typically the output of `get_qoi_gradient`.

        sigma_strategy: Defines the Sigma matrix (approximating the parameter covariance) to use.

            - 'identity': Use the identity matrix (Sigma = I).
                          Equivalent to the squared L2 norm of the gradients.

            - 'diagonal_empirical_fim': Calculate Sigma = diag( (diag(FIM) + damp)^-1 ).
                                        Requires `log_likelihood_functional`, `params`, `model`, `dataset`.
                                        `sigma_options` can contain `damping` (float) and `device`.

            - 'inverse_full_empirical_fim': Calculate Sigma = (FIM + damp * I)^-1.
                                             Requires `log_likelihood_functional`, `params`, `model`, `dataset`.
                                             `sigma_options` can contain `damping` (float), `device`,
                                             and `vmap_batch_size` (int).
                                             Warning: Computationally expensive (memory and time).

            - torch.Tensor (1D): Use these values directly as the *diagonal* of Sigma.
                                 Assumes Sigma is diagonal. Length must match flattened grad size.

            - torch.Tensor (2D): Use this tensor directly as the *full* Sigma matrix.
                                 Shape must be (P, P) where P is flattened grad size.

            - Callable: A function `f(flat_grads: torch.Tensor) -> scalar_tensor` that implements
                        the `grad^T * Sigma * grad` calculation with a custom Sigma application.

        sigma_options (dict, optional): Additional options for the chosen strategy.
                                       Common keys: 'damping' (float, for FIM), 'device' (torch.device),
                                       'vmap_batch_size' (int, optional, for FIM calculation).

        log_likelihood_functional: Required if sigma_strategy involves FIM calculation.
                                   Function `ll(params, model, data_point) -> scalar_log_likelihood`.
                                   `data_point` is an item yielded by `dataset`.

        params: Required if sigma_strategy involves FIM calculation. Model parameters PyTree.

        model: Required if sigma_strategy involves FIM calculation. The `nn.Module` model.

        dataset: Required if sigma_strategy involves FIM calculation. An iterable (e.g., DataLoader,
                 list) yielding data points compatible with `log_likelihood_functional`.

        device (torch.device, optional): The primary device to perform calculations on, especially
            for Sigma calculation if not provided directly. If None, inferred from parameters or gradients.

    Returns:
        A scalar tensor representing the calculated Delta Variance.

    Raises:
        ValueError: If required arguments for a sigma_strategy are missing, or if tensor
                    dimensions are incorrect.
        TypeError: If `sigma_strategy` type is unsupported.
    """
    sigma_options = sigma_options or {}
    flat_grads = flatten_pytree(qoi_gradient_pytree) # Get flat grads

    # Determine device if not explicitly provided
    if device is None:
        device = flat_grads.device
        logger.debug(f"Inferred device from gradients: {device}")

    # --- Argument Validation Helper --- #
    def _check_fim_args(strategy_name):
        missing = []
        if log_likelihood_functional is None: missing.append('log_likelihood_functional')
        if params is None: missing.append('params')
        if model is None: missing.append('model')
        if dataset is None: missing.append('dataset')
        if missing:
            raise ValueError(f"Missing required arguments for '{strategy_name}': {", ".join(missing)}")

    # --- Apply Sigma Strategy --- #
    if isinstance(sigma_strategy, str):
        strategy = sigma_strategy.lower() # Case-insensitive matching
        if strategy == 'identity':
            delta_var = apply_identity_sigma(flat_grads)

        elif strategy == 'diagonal_empirical_fim':
            _check_fim_args(strategy)
            damping = sigma_options.get('damping', 1e-5)
            calc_device = sigma_options.get('device', device)
            vmap_batch_size = sigma_options.get('vmap_batch_size', None)
            logger.info(f"Calculating inverse diagonal FIM (damping={damping}, device={calc_device})...")
            diag_sigma = calculate_inverse_fim_diagonal(
                log_likelihood_functional, params, model, dataset, damping=damping, device=calc_device,
                vmap_batch_size=vmap_batch_size
            )
            # Ensure diag_sigma is on the same device as gradients for the dot product
            delta_var = apply_diagonal_sigma(flat_grads, diag_sigma.to(flat_grads.device))

        elif strategy == 'inverse_full_empirical_fim':
            _check_fim_args(strategy)
            damping = sigma_options.get('damping', 1e-5)
            calc_device = sigma_options.get('device', device)
            vmap_batch_size = sigma_options.get('vmap_batch_size', None)
            logger.info(f"Calculating inverse full FIM (damping={damping}, device={calc_device})...")
            full_sigma = calculate_inverse_fim_full(
                log_likelihood_functional, params, model, dataset, damping=damping, device=calc_device,
                vmap_batch_size=vmap_batch_size
            )
            # Ensure full_sigma is on the same device as gradients for matmul
            delta_var = apply_full_sigma(flat_grads, full_sigma.to(flat_grads.device))

        else:
            raise ValueError(f"Unknown string sigma_strategy: {sigma_strategy}")

    elif isinstance(sigma_strategy, torch.Tensor):
        # Ensure sigma tensor is on the same device as gradients
        sigma_tensor = sigma_strategy.to(flat_grads.device)
        if sigma_tensor.dim() == 1:
            # User provided diagonal elements of Sigma directly
            logger.debug("Applying user-provided diagonal Sigma.")
            delta_var = apply_diagonal_sigma(flat_grads, sigma_tensor)
        elif sigma_tensor.dim() == 2:
            # User provided the full Sigma matrix directly
            logger.debug("Applying user-provided full Sigma matrix.")
            delta_var = apply_full_sigma(flat_grads, sigma_tensor)
        else:
            raise ValueError(f"Provided Sigma tensor must be 1D (diagonal) or 2D (full matrix), got shape {sigma_strategy.shape}")

    elif callable(sigma_strategy):
        # User provided a custom function to apply Sigma
        logger.debug("Applying user-provided custom Sigma function.")
        # Assuming the callable handles device placement internally if needed
        delta_var = sigma_strategy(flat_grads)
        if not isinstance(delta_var, torch.Tensor) or delta_var.numel() != 1:
            raise ValueError("Custom sigma function must return a scalar tensor.")

    else:
        raise TypeError(f"Unsupported sigma_strategy type: {type(sigma_strategy)}")

    return delta_var.squeeze() # Ensure scalar output
