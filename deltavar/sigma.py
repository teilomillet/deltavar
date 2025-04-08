import torch
from .utils import flatten_pytree, TensorTree
from typing import Callable, Union, Dict, Any, Iterable
from torch.func import grad, vmap
import logging # Use logging instead of print for warnings
import time # For timing FIM calculation
from itertools import islice
import math

# Setup basic logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- Sigma Application Strategies ---
# These functions take flattened gradients and apply a specific Sigma form

def apply_identity_sigma(flat_grads: torch.Tensor) -> torch.Tensor:
    """Applies Sigma = Identity. Delta Var = ||grads||^2."""
    return torch.dot(flat_grads, flat_grads)

def apply_diagonal_sigma(flat_grads: torch.Tensor, diag_elements: torch.Tensor) -> torch.Tensor:
    """Applies Sigma = diag(diag_elements). Delta Var = sum(grad_i^2 * sigma_i)."""
    if diag_elements.numel() != flat_grads.numel():
        raise ValueError(
            f"Diagonal Sigma size ({diag_elements.numel()}) must match "
            f"number of parameters ({flat_grads.numel()})"
        )
    if diag_elements.dim() != 1:
         raise ValueError(f"diag_elements must be a 1D tensor, got shape {diag_elements.shape}")
         
    # Ensure diag_elements is on the same device as flat_grads
    diag_elements = diag_elements.to(flat_grads.device)
    
    return torch.dot(flat_grads**2, diag_elements)

def apply_full_sigma(flat_grads: torch.Tensor, sigma_matrix: torch.Tensor) -> torch.Tensor:
    """Applies a full Sigma matrix. Delta Var = grads^T * Sigma * grads."""
    n_params = flat_grads.numel()
    if sigma_matrix.shape != (n_params, n_params):
        raise ValueError(
            f"Sigma matrix shape ({sigma_matrix.shape}) must be ({n_params}, {n_params})"
        )
    # Ensure sigma_matrix is on the same device as flat_grads
    sigma_matrix = sigma_matrix.to(flat_grads.device)
    
    # grads^T * Sigma * grads
    # Use matmul for efficiency: (1, N) @ (N, N) @ (N, 1) -> (1, 1)
    delta_var = torch.matmul(
        flat_grads.unsqueeze(0),
        torch.matmul(sigma_matrix, flat_grads.unsqueeze(-1))
    ).squeeze()
    return delta_var

# --- Sigma Calculation Strategies ---

def calculate_empirical_fim_diagonal(
    log_likelihood_functional: Callable, # log_likelihood(params, model, data_point) -> scalar
    params: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    dataset: Iterable[Any], # Expects an iterable yielding data points/batches
    device: torch.device = None,
    # Add option for vmap batch size - None means process all at once if possible
    vmap_batch_size: int = None
) -> torch.Tensor:
    """
    Calculates the diagonal elements of the Empirical Fisher Information Matrix (FIM).
    FIM_ii = E_data [ (grad_theta_i log f_theta(x))^2 ]

    Uses `torch.func.vmap` for potential speedup if the dataset can be batched.

    Args:
        log_likelihood_functional: Function `f(params, model, data_point) -> scalar_log_likelihood`.
                                   `data_point` should represent a SINGLE sample for vmap compatibility.
        params: Model parameters (PyTree, typically dict).
        model: The PyTorch model (nn.Module).
        dataset: An iterable (e.g., list, DataLoader) yielding single data points compatible
                 with `log_likelihood_functional`. Data points must be stackable if vmap_batch_size=None.
        device: The torch device to perform calculations on. If None, uses the device
                of the first parameter.
        vmap_batch_size (int, optional): If provided, process the dataset in batches of this size.
                                        Reduces peak memory usage but may be slightly slower overall.
                                        If None, attempts to process the entire dataset at once (requires stacking).

    Returns:
        A 1D tensor containing the diagonal elements of the FIM, flattened.
    """
    if device is None:
        first_param = next(iter(params.values())) # Basic inference
        device = first_param.device
        logger.debug(f"Inferred device: {device}")

    model.to(device)
    params_on_device = {k: v.to(device) for k, v in params.items()}

    # --- vmap approach --- 
    logger.info("Calculating Empirical FIM diagonal (using vmap approach)...")
    start_time = time.time()

    # 1. Define function to get *flattened* grad for a *single* sample
    def _get_flat_grad_single(p, single_data_point):
        # Ensure data_sample is on the correct device (essential for vmap)
        if isinstance(single_data_point, torch.Tensor):
            data_dev = single_data_point.to(device)
        elif isinstance(single_data_point, (list, tuple)): # Handle tuples like (input, target)
            data_dev = tuple(d.to(device) if isinstance(d, torch.Tensor) else d for d in single_data_point)
        else:
            data_dev = single_data_point # Assume non-tensor data doesn't need moving
            
        # Calculate per-sample grad using functional_call inside log_likelihood
        # Use grad w.r.t. first arg (p)
        grad_tree = grad(log_likelihood_functional, argnums=0)(p, model, data_dev)
        flat_grad = flatten_pytree(grad_tree)
        return flat_grad

    # 2. Create the vmapped gradient function (outside the loop/conditional)
    vmapped_grad_fn = vmap(_get_flat_grad_single, in_dims=(None, 0), randomness='error')

    num_params_flat = sum(p.numel() for p in params_on_device.values())
    total_samples = 0
    sum_sq_grads_flat = torch.zeros(num_params_flat, device=device)
    fim_diag_flat = torch.zeros(num_params_flat, device=device) # Initialize result

    if vmap_batch_size is None:
        # --- Process Full Dataset at Once ---
        logger.debug("vmap_batch_size is None. Attempting to stack and process the entire dataset.")
        try:
            dataset_list = list(dataset)
            num_samples = len(dataset_list)

            if num_samples > 0: # Only stack and compute if dataset is not empty
                # Assume dataset yields tuples (input, target) - adjust if format differs
                inputs = torch.stack([item[0] for item in dataset_list]).to(device)
                targets = torch.stack([item[1] for item in dataset_list]).to(device)
                batched_data = (inputs, targets)
                logger.debug(f"Stacked entire dataset ({num_samples} samples).")

                # Compute grads for the whole batch
                batch_flat_grads = vmapped_grad_fn(params_on_device, batched_data)
                # Output shape: (num_samples, num_params_flat)

                # Calculate FIM diagonal: E[grad^2] = mean(grad^2)
                fim_diag_flat = torch.mean(batch_flat_grads**2, dim=0)
                total_samples = num_samples # Set total samples processed
            # If num_samples is 0, total_samples remains 0, handled later

        except Exception as e:
            logger.error(f"Failed to stack or vmap the entire dataset. Consider using vmap_batch_size. Error: {e}")
            raise ValueError("Dataset could not be processed for vmap in one batch.") from e
    
    else:
        # --- Process Dataset in Chunks --- 
        logger.debug(f"Processing dataset in chunks of size {vmap_batch_size}.")
        dataset_iter = iter(dataset)
        processed_samples = 0
        while True:
            # Collect a chunk
            chunk = list(islice(dataset_iter, vmap_batch_size))
            if not chunk:
                break # End of dataset
                
            current_chunk_size = len(chunk)
            processed_samples += current_chunk_size
            
            try:
                # Stack the current chunk
                inputs = torch.stack([item[0] for item in chunk]).to(device)
                targets = torch.stack([item[1] for item in chunk]).to(device)
                batched_data = (inputs, targets)
                
                # Compute grads for the chunk
                chunk_flat_grads = vmapped_grad_fn(params_on_device, batched_data)
                # shape: (current_chunk_size, num_params_flat)
                
                # Accumulate SUM of squared gradients
                sum_sq_grads_flat += torch.sum(chunk_flat_grads**2, dim=0)
                
                if processed_samples % (vmap_batch_size * 10) == 0: # Log progress
                     logger.debug(f"Processed {processed_samples} samples...")

            except Exception as e:
                logger.error(f"Failed to stack or vmap chunk ending at sample approx {processed_samples}. Check data format within dataset. Error: {e}")
                # Decide whether to continue or raise
                raise ValueError("Failed processing dataset chunk.") from e
                
        total_samples = processed_samples
        if total_samples > 0:
            fim_diag_flat = sum_sq_grads_flat / total_samples

    # Final checks and logging
    if total_samples == 0:
        logger.warning("Dataset for FIM calculation was empty or could not be processed.")
        # Return zeros matching expected shape
        return torch.zeros(num_params_flat, device=device)
        
    end_time = time.time()
    processing_mode = f"batched ({vmap_batch_size})" if vmap_batch_size else "full batch"
    logger.info(f"Finished FIM diagonal calculation using vmap ({processing_mode}, {total_samples} samples) in {end_time - start_time:.2f} seconds.")

    return fim_diag_flat

def calculate_inverse_fim_diagonal(
    log_likelihood_functional: Callable,
    params: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    dataset: Any,
    damping: float = 1e-5, # Add damping for numerical stability before inversion
    device: torch.device = None,
    vmap_batch_size: int = None, # Pass batch size down
) -> torch.Tensor:
    """
    Calculates Sigma = diag( (FIM_diag + damping)^(-1) ).
    This is a common approximation where Sigma is the inverse of the *diagonal* FIM.

    Args:
        log_likelihood_functional, params, model, dataset: Passed to `calculate_empirical_fim_diagonal`.
        damping: Small value added to the diagonal before inversion for stability.
        device: The torch device for calculations.
        vmap_batch_size (int, optional): Batch size for vmap gradient calculation in `calculate_empirical_fim_diagonal`.

    Returns:
        A 1D tensor containing the diagonal elements of the approximate Sigma, flattened.
    """
    # Calculate the diagonal FIM, passing the batch size
    fim_diag = calculate_empirical_fim_diagonal(
        log_likelihood_functional, params, model, dataset, device=device, vmap_batch_size=vmap_batch_size
    )

    # Check for non-positive values which cause issues with inversion/sqrt
    if torch.any(fim_diag < 0):
        neg_count = torch.sum(fim_diag < 0).item()
        min_neg = torch.min(fim_diag[fim_diag < 0]).item() if neg_count > 0 else 0
        logger.warning(f"{neg_count} negative values found in FIM diagonal (min={min_neg:.4e}). Clamping to zero before adding damping.")
        fim_diag = torch.clamp(fim_diag, min=0)
        
    # Add damping and invert: sigma_ii = 1 / (FIM_ii + damping)
    inv_fim_diag = 1.0 / (fim_diag + damping)
    
    if torch.any(torch.isinf(inv_fim_diag)) or torch.any(torch.isnan(inv_fim_diag)):
         inf_count = torch.sum(torch.isinf(inv_fim_diag)).item()
         nan_count = torch.sum(torch.isnan(inv_fim_diag)).item()
         logger.warning(f"Infinity ({inf_count}) or NaN ({nan_count}) encountered in inverse FIM diagonal after damping. "
                        "This might indicate issues with gradients or very small FIM values near zero.")
         # Optionally handle NaNs/Infs, e.g., replace with a large number or zero?
         # For now, just warn.

    return inv_fim_diag

def calculate_empirical_fim_full(
    log_likelihood_functional: Callable,
    params: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    dataset: Iterable[Any],
    device: torch.device = None,
    # Add option for vmap batch size
    vmap_batch_size: int = None
) -> torch.Tensor:
    """
    Calculates the full Empirical Fisher Information Matrix (FIM).
    FIM = E_data [ flatten(grad_theta log f_theta(x)) @ flatten(grad_theta log f_theta(x)).T ]

    Uses `torch.func.vmap` for potential speedup if the dataset can be batched.

    Args:
        log_likelihood_functional: Function `f(params, model, data_point) -> scalar_log_likelihood`.
                                   `data_point` should represent a SINGLE sample for vmap compatibility.
        params: Model parameters (PyTree, typically dict).
        model: The PyTorch model (nn.Module).
        dataset: An iterable (e.g., list, DataLoader) yielding single data points compatible
                 with `log_likelihood_functional`. Data points must be stackable if vmap_batch_size=None.
        device: The torch device to perform calculations on. If None, uses the device
                of the first parameter.
        vmap_batch_size (int, optional): If provided, process the dataset in batches of this size.
                                        Reduces peak memory usage for gradient computation, but the
                                        full FIM matrix (P x P) still needs to be stored.
                                        If None, attempts to process the entire dataset at once (requires stacking).

    Returns:
        A 2D tensor representing the full FIM matrix (P x P).

    Warning: Requires O(P^2) memory. vmap helps with computation time but not memory peak for the matrix itself.
    """
    if device is None:
        first_param = next(iter(params.values())) # Basic inference
        device = first_param.device
        logger.debug(f"Inferred device: {device}")

    model.to(device)
    params_on_device = {k: v.to(device) for k, v in params.items()}

    num_params_flat = sum(p.numel() for p in params_on_device.values())
    logger.info(f"Calculating Full Empirical FIM for {num_params_flat} parameters (using vmap approach)... ")
    if num_params_flat > 20000: # Arbitrary threshold
         logger.warning(f"Number of parameters ({num_params_flat}) is large. Full FIM calculation might be very slow or run out of memory for the {num_params_flat}x{num_params_flat} matrix.")
    start_time = time.time()

    # 1. Define function to get *flattened* grad for a *single* sample (same as diagonal)
    def _get_flat_grad_single(p, single_data_point):
        # Ensure data_sample is on the correct device
        if isinstance(single_data_point, torch.Tensor):
            data_dev = single_data_point.to(device)
        elif isinstance(single_data_point, (list, tuple)):
            data_dev = tuple(d.to(device) if isinstance(d, torch.Tensor) else d for d in single_data_point)
        else:
            data_dev = single_data_point
        
        grad_tree = grad(log_likelihood_functional, argnums=0)(p, model, data_dev)
        flat_grad = flatten_pytree(grad_tree)
        return flat_grad

    # 2. Create the vmapped gradient function (outside loop/conditional)
    vmapped_grad_fn = vmap(_get_flat_grad_single, in_dims=(None, 0), randomness='error')

    total_samples = 0
    sum_outer_prods = torch.zeros((num_params_flat, num_params_flat), device=device)
    fim_full = torch.zeros((num_params_flat, num_params_flat), device=device) # Initialize result

    if vmap_batch_size is None:
        # --- Process Full Dataset at Once --- 
        logger.debug("vmap_batch_size is None. Attempting to stack and process the entire dataset.")
        try:
            dataset_list = list(dataset)
            num_samples = len(dataset_list)
            
            if num_samples > 0: # Only stack and compute if dataset is not empty
                # Assume dataset yields tuples (input, target) - adjust if format differs
                inputs = torch.stack([item[0] for item in dataset_list]).to(device)
                targets = torch.stack([item[1] for item in dataset_list]).to(device)
                batched_data = (inputs, targets)
                logger.debug(f"Stacked entire dataset ({num_samples} samples).")
                
                # Compute grads for the whole batch
                batch_flat_grads = vmapped_grad_fn(params_on_device, batched_data)
                # shape: (num_samples, num_params_flat)
                
                # Calculate FIM: E[gg^T] = (Sum gg^T)/N = (G^T G) / N
                fim_full = torch.matmul(batch_flat_grads.T, batch_flat_grads) / num_samples
                total_samples = num_samples
                # sum_outer_prods = fim_full * total_samples # Store sum for consistency - not needed here
            # If num_samples is 0, total_samples remains 0, handled later

        except Exception as e:
            logger.error(f"Failed to stack or vmap the entire dataset. Consider using vmap_batch_size. Error: {e}")
            raise ValueError("Dataset could not be processed for vmap in one batch.") from e

    else:
        # --- Process Dataset in Chunks ---
        logger.debug(f"Processing dataset in chunks of size {vmap_batch_size}.")
        dataset_iter = iter(dataset)
        processed_samples = 0
        while True:
            chunk = list(islice(dataset_iter, vmap_batch_size))
            if not chunk:
                break
                
            current_chunk_size = len(chunk)
            processed_samples += current_chunk_size

            try:
                inputs = torch.stack([item[0] for item in chunk]).to(device)
                targets = torch.stack([item[1] for item in chunk]).to(device)
                batched_data = (inputs, targets)
                
                # Compute grads for the chunk
                chunk_flat_grads = vmapped_grad_fn(params_on_device, batched_data)
                # shape: (current_chunk_size, num_params_flat)
                
                # Accumulate SUM of outer products: Sum( G_chunk.T @ G_chunk )
                sum_outer_prods += torch.matmul(chunk_flat_grads.T, chunk_flat_grads)
                
                if processed_samples % (vmap_batch_size * 10) == 0:
                    logger.debug(f"Processed {processed_samples} samples...")

            except Exception as e:
                logger.error(f"Failed to stack or vmap chunk ending at sample approx {processed_samples}. Error: {e}")
                raise ValueError("Failed processing dataset chunk.") from e

        total_samples = processed_samples
        if total_samples > 0:
            fim_full = sum_outer_prods / total_samples

    # Final checks and logging
    if total_samples == 0:
        logger.warning("Dataset for FIM calculation was empty or could not be processed.")
         # Return zeros matching expected shape
        return torch.zeros((num_params_flat, num_params_flat), device=device)

    end_time = time.time()
    processing_mode = f"batched ({vmap_batch_size})" if vmap_batch_size else "full batch"
    logger.info(f"Finished full FIM calculation using vmap ({processing_mode}, {total_samples} samples) in {end_time - start_time:.2f} seconds.")
    
    # --- Robustness Check: Check for negative eigenvalues --- 
    # FIM should be positive semi-definite. Negative eigenvalues indicate numerical issues.
    # Use a tolerance as small negative values can occur due to floating point errors.
    try:
        with torch.no_grad(): # Avoid tracking grads during check
            eigenvalues = torch.linalg.eigvalsh(fim_full)
            min_eigenvalue = torch.min(eigenvalues).item()
            if min_eigenvalue < -1e-6: # Allow for small numerical errors
                logger.warning(
                    f"Calculated FIM has significantly negative eigenvalues (min={min_eigenvalue:.4e}). "
                    f"This suggests potential numerical instability. Damping will be applied before inversion, "
                    f"but the FIM estimate might be unreliable."
                )
            elif min_eigenvalue < 0:
                 logger.debug(f"Calculated FIM has small negative eigenvalues (min={min_eigenvalue:.4e}), likely due to numerical precision. Proceeding with damping.")
    except torch.linalg.LinAlgError as e:
         logger.warning(f"Could not compute eigenvalues for FIM check: {e}")
    # --- End Robustness Check ---

    # Optional: Check symmetry (should be symmetric by construction)
    # if not torch.allclose(fim_full, fim_full.T, atol=1e-5):

    return fim_full

def calculate_inverse_fim_full(
    log_likelihood_functional: Callable,
    params: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    dataset: Any,
    damping: float = 1e-5,
    device: torch.device = None,
    vmap_batch_size: int = None, # Pass batch size down
) -> torch.Tensor:
    """
    Calculates Sigma = (FIM + damping * I)^(-1).

    Args:
        log_likelihood_functional, params, model, dataset, device: Passed to `calculate_empirical_fim_full`.
        damping: Small value added to the diagonal before inversion for stability.
        vmap_batch_size (int, optional): Batch size for vmap gradient calculation in `calculate_empirical_fim_full`.

    Returns:
        A 2D tensor representing the inverse of the (damped) full FIM matrix (P x P).

    Warning: Requires O(P^3) computation for inversion.
    """
    # Calculate the full FIM, passing the batch size
    fim_full = calculate_empirical_fim_full(
        log_likelihood_functional, params, model, dataset, device=device, vmap_batch_size=vmap_batch_size
    )

    num_params_flat = fim_full.shape[0]
    if num_params_flat == 0:
         logger.warning("FIM matrix is empty (0x0). Returning empty tensor.")
         return torch.empty((0,0), device=fim_full.device)

    # Add damping to the diagonal: F + lambda * I
    damped_fim = fim_full + torch.eye(num_params_flat, device=fim_full.device) * damping

    logger.info(f"Inverting damped {num_params_flat}x{num_params_flat} FIM matrix...")
    start_time = time.time()
    try:
        # Compute the inverse
        # Using torch.linalg.inv for better numerical stability than .inverse()
        inv_fim_full = torch.linalg.inv(damped_fim)
        end_time = time.time()
        logger.info(f"Finished FIM inversion in {end_time - start_time:.2f} seconds.")
    except torch.linalg.LinAlgError as e:
        logger.error(f"Failed to invert the FIM matrix: {e}")
        logger.error("The damped FIM matrix might still be singular or ill-conditioned.")
        # Handle error: maybe return None, raise exception, or return pseudo-inverse?
        # For now, re-raise the exception.
        raise e
        # Alternative: Use pseudo-inverse (pinv)
        # logger.warning("Using pseudo-inverse due to inversion error.")
        # inv_fim_full = torch.linalg.pinv(damped_fim)

    if torch.any(torch.isinf(inv_fim_full)) or torch.any(torch.isnan(inv_fim_full)):
         inf_count = torch.sum(torch.isinf(inv_fim_full)).item()
         nan_count = torch.sum(torch.isnan(inv_fim_full)).item()
         logger.warning(f"Infinity ({inf_count}) or NaN ({nan_count}) encountered in the inverse FIM matrix.")

    return inv_fim_full
