import torch
import torch.nn as nn
from torch.func import functional_call
from collections import OrderedDict
import sys
import os
import time
import logging # Import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)

# Add package parent directory to path if running script directly
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from the package
try:
    from deltavar import get_qoi_gradient, calculate_delta_variance
except ImportError:
    logger.error("Error: Could not import deltavar. Ensure the package is installed or path is set.")
    sys.exit(1)


# --- Define a User Model ---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = OrderedDict()
        in_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers[f'linear_{i}'] = nn.Linear(in_dim, h_dim)
            layers[f'relu_{i}'] = nn.ReLU()
            in_dim = h_dim
        layers['output'] = nn.Linear(in_dim, output_dim)
        self.net = nn.Sequential(layers)

    def forward(self, x):
        return self.net(x)

# --- Define the Quantity of Interest (QoI) Functional ---
# This function takes params, model, input -> scalar tensor
def qoi_sum_output(params, model, z):
    """Example QoI: Sum of the model's output vector."""
    # Ensure z has batch dim if needed by model forward
    if z.dim() == 1:
        z_batch = z.unsqueeze(0)
    else:
        z_batch = z
        
    output = functional_call(model, params, (z_batch,))
    
    # Return a scalar value
    return output.sum()

def qoi_first_output(params, model, z):
    """Example QoI: First element of the model's output vector."""
    if z.dim() == 1:
        z_batch = z.unsqueeze(0)
    else:
        z_batch = z
    output = functional_call(model, params, (z_batch,))
    # Return a scalar value
    return output.squeeze()[0] # Assume first output of first batch item

# --- Define Log-Likelihood Functional (for FIM) ---
# Example: Assume Gaussian likelihood with fixed variance for regression
def log_likelihood_gaussian(params, model, data_point):
    """Example Log Likelihood: Gaussian N(output | model(input), sigma^2)."""
    # Assumes data_point is a tuple (input, target)
    input_x, target_y = data_point
    if input_x.dim() == 1:
        input_x = input_x.unsqueeze(0)
        
    prediction = functional_call(model, params, (input_x,))
    
    # Assume output_dim matches target_y.shape[-1]
    # Use a fixed std deviation for simplicity
    std_dev = 1.0
    # Gaussian log prob formula: -0.5 * [ log(2*pi*sigma^2) + ((y - mu)/sigma)^2 ]
    log_prob = -0.5 * (torch.log(torch.tensor(2.0 * torch.pi)) + 2 * torch.log(torch.tensor(std_dev)) + \
                       ((target_y - prediction) / std_dev)**2)
                       
    return log_prob.sum() # Sum over output dimensions if multi-output


if __name__ == '__main__':
    # --- Configuration ---
    input_dim = 5 # Reduced for faster full FIM demo
    hidden_dims = [10, 8] # Reduced for faster full FIM demo
    output_dim = 2 # Reduced for faster full FIM demo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Model and Parameters ---
    model = SimpleMLP(input_dim, hidden_dims, output_dim).to(device)
    # Get parameters as a dictionary. DO NOT modify the model directly after this.
    # The functional API requires static parameters.
    params = {k: v.detach().clone() for k, v in model.named_parameters()} # Work with copies
    num_params = sum(p.numel() for p in params.values())
    logger.info(f"Model has {num_params} parameters.")

    # --- Example Input ---
    z = torch.randn(input_dim, device=device)

    # --- Calculate QoI Gradient ---
    logger.info("\nCalculating QoI gradient for 'sum_output'...")
    qoi_grad_sum = get_qoi_gradient(qoi_sum_output, params, model, z)
    # print("Gradient structure (first layer weight):", qoi_grad_sum['net.linear_0.weight'].shape)

    logger.info("Calculating QoI gradient for 'first_output'...")
    qoi_grad_first = get_qoi_gradient(qoi_first_output, params, model, z)


    # --- Create Dummy Dataset ---
    # Reduced size for faster demo
    dataset_size = 50
    logger.info(f"Creating dummy dataset (size={dataset_size})...")
    # Ensure data is created on the correct device from the start or moved later
    dummy_dataset = [(torch.randn(input_dim, device=device), torch.randn(output_dim, device=device))
                     for _ in range(dataset_size)]

    # --- Calculate Delta Variance ---
    damping_value = 1e-4

    # 1. Identity Sigma
    logger.info("\nCalculating Delta Variance (Sigma = Identity)")
    start = time.time()
    dv_identity_sum = calculate_delta_variance(qoi_grad_sum, sigma_strategy='identity', device=device)
    logger.info(f"  DeltaVar (QoI=sum, Sigma=I): {dv_identity_sum.item():.4f} [Time: {time.time()-start:.2f}s]")

    # 2. Diagonal FIM Approx
    logger.info("\nCalculating Delta Variance (Sigma = Inv Diag FIM)")
    start = time.time()
    dv_fim_diag_sum = calculate_delta_variance(
        qoi_gradient_pytree=qoi_grad_sum,
        sigma_strategy='diagonal_empirical_fim',
        log_likelihood_functional=log_likelihood_gaussian,
        params=params, model=model, dataset=dummy_dataset,
        sigma_options={'damping': damping_value},
        device=device
    )
    logger.info(f"  DeltaVar (QoI=sum, Sigma=InvDiagFIM): {dv_fim_diag_sum.item():.4f} [Time: {time.time()-start:.2f}s]")


    # 3. Full FIM Approx (Potentially Slow!)
    logger.info("\nCalculating Delta Variance (Sigma = Inv Full FIM) - MIGHT BE SLOW")
    start = time.time()
    try:
        dv_fim_full_sum = calculate_delta_variance(
            qoi_gradient_pytree=qoi_grad_sum,
            sigma_strategy='inverse_full_empirical_fim',
            log_likelihood_functional=log_likelihood_gaussian,
            params=params, model=model, dataset=dummy_dataset,
            sigma_options={'damping': damping_value},
            device=device
        )
        logger.info(f"  DeltaVar (QoI=sum, Sigma=InvFullFIM): {dv_fim_full_sum.item():.4f} [Time: {time.time()-start:.2f}s]")
    except torch.linalg.LinAlgError as e:
         logger.error(f"  Failed to calculate DeltaVar with full FIM: {e}")
    except Exception as e: # Catch other potential errors like OOM
        logger.error(f"  An unexpected error occurred during full FIM calculation: {e}")

    logger.info("\nExample finished.")
