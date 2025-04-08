import pytest
import torch
import torch.nn as nn
from torch.func import functional_call
from collections import OrderedDict
import sys
import os

# Add package parent directory to path if running tests directly
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from the package (assuming deltavar/core.py exists)
try:
    from deltavar.core import (
        get_qoi_gradient,
        calculate_delta_variance,
        # Internal FIM functions are tested via calculate_delta_variance
        # _calculate_empirical_fim_diagonal, # Removed
        # _calculate_empirical_fim_full,     # Removed
    )
except ImportError as e:
    # Make the error message more informative
    pytest.skip(f"Cannot import deltavar.core ({e}), skipping tests.", allow_module_level=True)

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

# --- Test Fixtures ---

# Fixture for simple linear model/QoI for gradient check
@pytest.fixture(scope="module")
def simple_linear_setup():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(1, 1, bias=False).to(device)
    # y = w*x
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0]], device=device))
    params = {k: v.detach().clone() for k, v in model.named_parameters()}
    z = torch.tensor([3.0], device=device) # Input x

    # QoI = output y
    def qoi_output(params, model, z):
        # Ensure z has batch dim for model
        if z.dim() == 0: z_batch = z.unsqueeze(0).unsqueeze(-1) # scalar to (1,1)
        elif z.dim() == 1: z_batch = z.unsqueeze(0) # vector to (1, n)
        else: z_batch = z
        output = functional_call(model, params, (z_batch,))
        return output.sum() # sum needed for scalar output for grad

    return {"model": model, "params": params, "z": z, "qoi_func": qoi_output, "device": device}

@pytest.fixture(scope="module")
def setup_data():
    """Provides a consistent setup for model, params, data, and funcs."""
    input_dim = 3
    hidden_dims = [5]
    output_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleMLP(input_dim, hidden_dims, output_dim).to(device)
    # Use fixed seeds for reproducibility in tests
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    params = {k: v.detach().clone().requires_grad_(False) for k, v in model.named_parameters()}
    num_params = sum(p.numel() for p in params.values())

    # Single input point for QoI grad
    z = torch.randn(input_dim, device=device)

    # Dummy dataset for FIM
    dataset_size = 10
    dummy_dataset = [(torch.randn(input_dim, device=device), torch.randn(output_dim, device=device))
                     for _ in range(dataset_size)]

    # --- Define QoI Functional ---
    def qoi_sum_output(params, model, z):
        if z.dim() == 1: z_batch = z.unsqueeze(0)
        else: z_batch = z
        output = functional_call(model, params, (z_batch,))
        return output.sum()

    # --- Define Log-Likelihood Functional ---
    def log_likelihood_gaussian(params, model, data_point):
        input_x, target_y = data_point
        if input_x.dim() == 1: input_x = input_x.unsqueeze(0)
        prediction = functional_call(model, params, (input_x,))
        std_dev = 1.0
        log_prob = -0.5 * (torch.log(torch.tensor(2.0 * torch.pi)) + 2 * torch.log(torch.tensor(std_dev)) + \
                           ((target_y - prediction) / std_dev)**2)
        return log_prob.sum()

    return {
        "model": model,
        "params": params,
        "z": z,
        "dataset": dummy_dataset,
        "qoi_func": qoi_sum_output,
        "log_likelihood_func": log_likelihood_gaussian,
        "device": device,
        "num_params": num_params
    }

# --- Test Functions ---

def test_get_qoi_gradient(setup_data):
    """Verify the shape and basic properties of the QoI gradient."""
    qoi_grad = get_qoi_gradient(
        setup_data["qoi_func"],
        setup_data["params"],
        setup_data["model"],
        setup_data["z"]
    )
    assert isinstance(qoi_grad, dict)
    assert qoi_grad.keys() == setup_data["params"].keys()
    for name, grad_tensor in qoi_grad.items():
        assert grad_tensor.shape == setup_data["params"][name].shape
        assert grad_tensor.device == setup_data["device"]
    # Optional: Add a check for non-zero gradients if expected

def test_get_qoi_gradient_correctness(simple_linear_setup):
    """Verify get_qoi_gradient calculation for a simple linear case."""
    qoi_grad = get_qoi_gradient(
        simple_linear_setup["qoi_func"],
        simple_linear_setup["params"],
        simple_linear_setup["model"],
        simple_linear_setup["z"]
    )
    # Model: y = w*x. QoI = y. grad_w(QoI) = x
    expected_grad = simple_linear_setup["z"]
    # The gradient dict contains {'weight': tensor([[grad]])}
    assert torch.allclose(qoi_grad['weight'].squeeze(), expected_grad)
    assert qoi_grad['weight'].device == simple_linear_setup["device"]

def test_calculate_delta_variance_identity(setup_data):
    """Test delta variance calculation with identity sigma."""
    qoi_grad = get_qoi_gradient(
        setup_data["qoi_func"], setup_data["params"], setup_data["model"], setup_data["z"]
    )
    dv = calculate_delta_variance(
        qoi_gradient_pytree=qoi_grad,
        sigma_strategy='identity',
        device=setup_data["device"]
    )
    assert isinstance(dv, torch.Tensor)
    assert dv.ndim == 0 # Should be scalar
    assert dv.item() > 0 # Variance should be positive
    # Check calculation: sum of squared gradients
    expected_dv = sum(torch.sum(g**2) for g in qoi_grad.values())
    assert torch.allclose(dv, expected_dv)
    assert dv.device == setup_data["device"]

def test_calculate_delta_variance_diag_fim(setup_data):
    """Test delta variance with diagonal FIM sigma."""
    qoi_grad = get_qoi_gradient(
        setup_data["qoi_func"], setup_data["params"], setup_data["model"], setup_data["z"]
    )
    damping = 1e-4
    dv = calculate_delta_variance(
        qoi_gradient_pytree=qoi_grad,
        sigma_strategy='diagonal_empirical_fim',
        log_likelihood_functional=setup_data["log_likelihood_func"],
        params=setup_data["params"],
        model=setup_data["model"],
        dataset=setup_data["dataset"],
        sigma_options={'damping': damping, 'device': setup_data["device"]},
    )
    assert isinstance(dv, torch.Tensor)
    assert dv.ndim == 0
    assert dv.item() > 0
    # Optional: Add a check against manual calculation if feasible
    assert dv.device == setup_data["device"]

def test_calculate_delta_variance_diag_fim_batched(setup_data):
    """Test diagonal FIM with vmap_batch_size."""
    qoi_grad = get_qoi_gradient(
        setup_data["qoi_func"], setup_data["params"], setup_data["model"], setup_data["z"]
    )
    damping = 1e-4
    batch_size = len(setup_data["dataset"]) // 2 # Example batch size
    dv = calculate_delta_variance(
        qoi_gradient_pytree=qoi_grad,
        sigma_strategy='diagonal_empirical_fim',
        log_likelihood_functional=setup_data["log_likelihood_func"],
        params=setup_data["params"],
        model=setup_data["model"],
        dataset=setup_data["dataset"],
        sigma_options={'damping': damping, 'device': setup_data["device"], 'vmap_batch_size': batch_size},
    )
    assert isinstance(dv, torch.Tensor)
    assert dv.ndim == 0
    assert dv.device == setup_data["device"]
    assert dv.item() > 0
    # Ideally, compare dv with the unbatched version, should be very close
    dv_unbatched = calculate_delta_variance(
        qoi_gradient_pytree=qoi_grad,
        sigma_strategy='diagonal_empirical_fim',
        log_likelihood_functional=setup_data["log_likelihood_func"],
        params=setup_data["params"],
        model=setup_data["model"],
        dataset=setup_data["dataset"],
        sigma_options={'damping': damping, 'device': setup_data["device"]},
    )
    assert torch.allclose(dv, dv_unbatched, atol=1e-5)

def test_calculate_delta_variance_full_fim(setup_data):
    """Test delta variance with full FIM sigma."""
    qoi_grad = get_qoi_gradient(
        setup_data["qoi_func"], setup_data["params"], setup_data["model"], setup_data["z"]
    )
    damping = 1e-4
    dv = calculate_delta_variance(
        qoi_gradient_pytree=qoi_grad,
        sigma_strategy='inverse_full_empirical_fim',
        log_likelihood_functional=setup_data["log_likelihood_func"],
        params=setup_data["params"],
        model=setup_data["model"],
        dataset=setup_data["dataset"],
        sigma_options={'damping': damping, 'device': setup_data["device"]},
    )
    assert isinstance(dv, torch.Tensor)
    assert dv.ndim == 0
    assert dv.item() > 0
    # Optional: Add a check against manual calculation if feasible 
    assert dv.device == setup_data["device"]

def test_calculate_delta_variance_full_fim_batched(setup_data):
    """Test full FIM with vmap_batch_size."""
    qoi_grad = get_qoi_gradient(
        setup_data["qoi_func"], setup_data["params"], setup_data["model"], setup_data["z"]
    )
    damping = 1e-4
    batch_size = len(setup_data["dataset"]) // 2 # Example batch size
    dv = calculate_delta_variance(
        qoi_gradient_pytree=qoi_grad,
        sigma_strategy='inverse_full_empirical_fim',
        log_likelihood_functional=setup_data["log_likelihood_func"],
        params=setup_data["params"],
        model=setup_data["model"],
        dataset=setup_data["dataset"],
        sigma_options={'damping': damping, 'device': setup_data["device"], 'vmap_batch_size': batch_size},
    )
    assert isinstance(dv, torch.Tensor)
    assert dv.ndim == 0
    assert dv.device == setup_data["device"]
    assert dv.item() > 0
    # Compare with unbatched version
    dv_unbatched = calculate_delta_variance(
        qoi_gradient_pytree=qoi_grad,
        sigma_strategy='inverse_full_empirical_fim',
        log_likelihood_functional=setup_data["log_likelihood_func"],
        params=setup_data["params"],
        model=setup_data["model"],
        dataset=setup_data["dataset"],
        sigma_options={'damping': damping, 'device': setup_data["device"]},
    )
    assert torch.allclose(dv, dv_unbatched, rtol=1e-3)

def test_calculate_delta_variance_preset_diag(setup_data):
    """Test delta variance with a preset diagonal sigma tensor."""
    qoi_grad = get_qoi_gradient(
        setup_data["qoi_func"], setup_data["params"], setup_data["model"], setup_data["z"]
    )
    num_params = setup_data["num_params"]
    # Create a dummy diagonal sigma
    preset_diag_sigma = torch.ones(num_params, device=setup_data["device"]) * 0.5

    dv = calculate_delta_variance(
        qoi_gradient_pytree=qoi_grad,
        sigma_strategy=preset_diag_sigma,
        device=setup_data["device"]
    )
    assert isinstance(dv, torch.Tensor)
    assert dv.ndim == 0
    assert dv.item() >= 0 # Can be zero if grads are zero

    # Check calculation: sum(grad_i^2 * sigma_i)
    from deltavar.utils import flatten_pytree
    flat_grads = flatten_pytree(qoi_grad)
    expected_dv = torch.dot(flat_grads**2, preset_diag_sigma)
    assert torch.allclose(dv, expected_dv)
    assert dv.device == setup_data["device"]

def test_calculate_delta_variance_preset_full(setup_data):
    """Test delta variance with a preset full sigma tensor."""
    qoi_grad = get_qoi_gradient(
        setup_data["qoi_func"], setup_data["params"], setup_data["model"], setup_data["z"]
    )
    num_params = setup_data["num_params"]
    device = setup_data["device"]
    # Create a dummy full sigma (identity for simplicity)
    preset_full_sigma = torch.eye(num_params, device=device) * 0.5

    dv = calculate_delta_variance(
        qoi_gradient_pytree=qoi_grad,
        sigma_strategy=preset_full_sigma,
        device=device
    )
    assert isinstance(dv, torch.Tensor)
    assert dv.ndim == 0
    assert dv.item() >= 0

    # Check calculation: grad^T * Sigma * grad
    from deltavar.utils import flatten_pytree
    flat_grads = flatten_pytree(qoi_grad)
    expected_dv = flat_grads.unsqueeze(0) @ preset_full_sigma @ flat_grads.unsqueeze(-1)
    assert torch.allclose(dv, expected_dv.squeeze())
    assert dv.device == setup_data["device"]

def test_calculate_delta_variance_callable_sigma(setup_data):
    """Test delta variance with a callable sigma strategy."""
    qoi_grad = get_qoi_gradient(
        setup_data["qoi_func"], setup_data["params"], setup_data["model"], setup_data["z"]
    )
    device = setup_data["device"]

    # Define a simple callable (e.g., scales identity)
    scale_factor = 2.5
    def custom_sigma_apply(flat_grads):
        # Example: Equivalent to Sigma = scale_factor * Identity
        return scale_factor * torch.dot(flat_grads, flat_grads)

    dv = calculate_delta_variance(
        qoi_gradient_pytree=qoi_grad,
        sigma_strategy=custom_sigma_apply, # Pass the function directly
        device=device
    )
    assert isinstance(dv, torch.Tensor)
    assert dv.ndim == 0
    assert dv.item() >= 0

    # Check calculation matches the custom function
    from deltavar.utils import flatten_pytree
    flat_grads = flatten_pytree(qoi_grad)
    expected_dv = scale_factor * torch.dot(flat_grads, flat_grads)
    assert torch.allclose(dv, expected_dv)
    assert dv.device == setup_data["device"]

def test_calculate_delta_variance_missing_args(setup_data):
    """Test ValueError when required args for FIM are missing."""
    qoi_grad = get_qoi_gradient(
        setup_data["qoi_func"], setup_data["params"], setup_data["model"], setup_data["z"]
    )
    # Test missing args for diagonal FIM
    with pytest.raises(ValueError, match="Missing required arguments for 'diagonal_empirical_fim'"):
        calculate_delta_variance(qoi_grad, sigma_strategy='diagonal_empirical_fim',
                                 params=setup_data["params"], model=setup_data["model"], dataset=setup_data["dataset"])
    with pytest.raises(ValueError, match="Missing required arguments for 'diagonal_empirical_fim'"):
        calculate_delta_variance(qoi_grad, sigma_strategy='diagonal_empirical_fim',
                                 log_likelihood_functional=setup_data["log_likelihood_func"], model=setup_data["model"], dataset=setup_data["dataset"])
    # Test missing args for full FIM
    with pytest.raises(ValueError, match="Missing required arguments for 'inverse_full_empirical_fim'"):
        calculate_delta_variance(qoi_grad, sigma_strategy='inverse_full_empirical_fim',
                                 log_likelihood_functional=setup_data["log_likelihood_func"], params=setup_data["params"], dataset=setup_data["dataset"])

def test_calculate_delta_variance_error_handling(setup_data):
    """Test error handling for bad sigma strategy inputs."""
    qoi_grad = get_qoi_gradient(
        setup_data["qoi_func"], setup_data["params"], setup_data["model"], setup_data["z"]
    )
    num_params = setup_data["num_params"]
    device = setup_data["device"]

    # Bad string strategy
    with pytest.raises(ValueError, match="Unknown string sigma_strategy"):
        calculate_delta_variance(qoi_grad, sigma_strategy='invalid_strategy', device=device)

    # Bad tensor shapes
    wrong_diag_shape = torch.ones(num_params + 1, device=device) # Wrong size
    with pytest.raises(ValueError, match="Diagonal Sigma size"):
        calculate_delta_variance(qoi_grad, sigma_strategy=wrong_diag_shape, device=device)

    wrong_full_shape = torch.eye(num_params + 1, device=device) # Wrong size
    with pytest.raises(ValueError, match="Sigma matrix shape"):
        calculate_delta_variance(qoi_grad, sigma_strategy=wrong_full_shape, device=device)

    bad_dim_tensor = torch.randn(num_params, 1, 1, device=device) # Wrong dimensions (3D)
    with pytest.raises(ValueError, match="Provided Sigma tensor must be 1D .* or 2D"):
        calculate_delta_variance(qoi_grad, sigma_strategy=bad_dim_tensor, device=device)

    # Bad callable output
    def bad_callable(flat_grads): return torch.randn(2, device=flat_grads.device)
    with pytest.raises(ValueError, match="Custom sigma function must return a scalar tensor"):
        calculate_delta_variance(qoi_grad, sigma_strategy=bad_callable, device=device)

def test_calculate_delta_variance_zero_gradient(setup_data):
    """Test delta variance calculation when QoI gradient is zero."""
    # Define QoI that is constant w.r.t params
    def qoi_constant(params, model, z): return torch.tensor(1.0)

    zero_qoi_grad = get_qoi_gradient(
        qoi_constant,
        setup_data["params"],
        setup_data["model"],
        setup_data["z"]
    )

    # Check grads are actually zero
    for name, grad_tensor in zero_qoi_grad.items():
        assert torch.all(grad_tensor == 0)

    # Test with identity sigma
    dv_identity = calculate_delta_variance(
        qoi_gradient_pytree=zero_qoi_grad,
        sigma_strategy='identity',
        device=setup_data["device"]
    )
    assert dv_identity == 0.0
    assert dv_identity.device == setup_data["device"]

    # Test with preset diagonal sigma
    num_params = setup_data["num_params"]
    preset_diag_sigma = torch.rand(num_params, device=setup_data["device"]) # Use random positive values
    dv_preset_diag = calculate_delta_variance(
        qoi_gradient_pytree=zero_qoi_grad,
        sigma_strategy=preset_diag_sigma,
        device=setup_data["device"]
    )
    assert dv_preset_diag == 0.0
    assert dv_preset_diag.device == setup_data["device"]

    # Test with diag FIM (should also be zero)
    # Note: FIM itself isn't zero, but grad^T * FIM^-1 * grad is
    damping = 1e-4
    dv_diag_fim = calculate_delta_variance(
        qoi_gradient_pytree=zero_qoi_grad,
        sigma_strategy='diagonal_empirical_fim',
        log_likelihood_functional=setup_data["log_likelihood_func"],
        params=setup_data["params"],
        model=setup_data["model"],
        dataset=setup_data["dataset"],
        sigma_options={'damping': damping, 'device': setup_data["device"]},
    )
    assert dv_diag_fim == 0.0
    assert dv_diag_fim.device == setup_data["device"]

# TODO: Remaining?
# - Test device handling more explicitly (e.g., sigma_options device vs grad device)
# (Current tests cover basic device consistency) 