import pytest
import torch
from torch.func import functional_call # Import functional_call
from deltavar.sigma import (
    apply_identity_sigma,
    apply_diagonal_sigma,
    apply_full_sigma,
    calculate_empirical_fim_diagonal,
    calculate_inverse_fim_diagonal,
    calculate_empirical_fim_full,
    calculate_inverse_fim_full,
)
from deltavar.utils import flatten_pytree # Assuming utils is needed later
import torch.nn as nn

# Fixtures for basic test data
@pytest.fixture
def flat_grads_small():
    """Simple flat gradient tensor."""
    return torch.tensor([1.0, 2.0, -3.0])

@pytest.fixture
def diag_elements_small():
    """Matching diagonal elements for flat_grads_small."""
    return torch.tensor([0.5, 1.0, 2.0])

@pytest.fixture
def full_sigma_small():
    """Matching full sigma matrix for flat_grads_small."""
    # Example: A simple positive definite matrix
    return torch.tensor([
        [2.0, 0.1, 0.0],
        [0.1, 1.0, -0.2],
        [0.0, -0.2, 3.0]
    ])

# === Tests for Sigma Application Functions ===

def test_apply_identity_sigma(flat_grads_small):
    """Test Sigma=I application."""
    expected_dot_product = torch.dot(flat_grads_small, flat_grads_small)
    assert torch.allclose(apply_identity_sigma(flat_grads_small), expected_dot_product)

def test_apply_diagonal_sigma(flat_grads_small, diag_elements_small):
    """Test diagonal Sigma application."""
    expected_value = torch.dot(flat_grads_small**2, diag_elements_small)
    assert torch.allclose(apply_diagonal_sigma(flat_grads_small, diag_elements_small), expected_value)

def test_apply_diagonal_sigma_shape_mismatch(flat_grads_small):
    """Test error handling for diagonal shape mismatch."""
    wrong_diag = torch.tensor([1.0, 2.0]) # Incorrect size
    with pytest.raises(ValueError, match="Diagonal Sigma size"):
        apply_diagonal_sigma(flat_grads_small, wrong_diag)
    wrong_diag_2d = torch.tensor([[1.0],[2.0],[3.0]]) # Incorrect dim
    with pytest.raises(ValueError, match="diag_elements must be a 1D tensor"):
        apply_diagonal_sigma(flat_grads_small, wrong_diag_2d)

def test_apply_full_sigma(flat_grads_small, full_sigma_small):
    """Test full Sigma application."""
    grads = flat_grads_small
    sigma = full_sigma_small
    # Expected: g^T * S * g
    expected_value = grads.unsqueeze(0) @ sigma @ grads.unsqueeze(-1)
    assert torch.allclose(apply_full_sigma(grads, sigma), expected_value.squeeze())

def test_apply_full_sigma_shape_mismatch(flat_grads_small):
    """Test error handling for full sigma shape mismatch."""
    wrong_sigma = torch.tensor([[1.0, 0.0], [0.0, 1.0]]) # Wrong shape (2x2 vs 3)
    with pytest.raises(ValueError, match="Sigma matrix shape"):
        apply_full_sigma(flat_grads_small, wrong_sigma)
    wrong_sigma_3d = torch.randn(3, 3, 3) # Wrong dimensions
    with pytest.raises(ValueError, match="Sigma matrix shape"):
        apply_full_sigma(flat_grads_small, wrong_sigma_3d)

# === Fixtures for FIM Tests ===

@pytest.fixture
def simple_linear_model():
    """A simple linear model for testing."""
    model = nn.Linear(2, 1, bias=False) # Simple: y = w1*x1 + w2*x2
    # Initialize weights for reproducibility
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.5, -1.5]])) # Shape (out_features, in_features)
    return model

@pytest.fixture
def simple_params(simple_linear_model):
    """Extract parameters from the simple model."""
    return {name: p.detach().clone() for name, p in simple_linear_model.named_parameters()}

@pytest.fixture
def simple_dataset():
    """A small dataset (list of tuples) for the linear model."""
    # Data points are (input_tensor, target_tensor)
    # Inputs are shape (1, 2), Targets are shape (1, 1)
    return [
        (torch.tensor([[1.0, 1.0]]), torch.tensor([[1.0]])),  # x = [1, 1], y = 1
        (torch.tensor([[2.0, 0.0]]), torch.tensor([[1.5]])),  # x = [2, 0], y = 1.5
        (torch.tensor([[0.0, -1.0]]), torch.tensor([[-1.0]])) # x = [0, -1], y = -1
    ]

def simple_gaussian_nll(params, model, data_point):
    """Gaussian Negative Log Likelihood (assuming variance=1)."""
    # Assumes model is passed correctly and used via functional_call
    # Assumes data_point is a tuple (input, target)
    x, y_true = data_point
    # Use functional_call for stateless computation
    y_pred = functional_call(model, params, (x,))
    # NLL = 0.5 * (y_true - y_pred)**2 + const (ignore const)
    loss = 0.5 * torch.mean((y_true - y_pred)**2) # Use mean for scalar output
    return loss

# === Tests for FIM Calculation Functions ===

def test_calculate_fim_diagonal_smoke(simple_params, simple_linear_model, simple_dataset):
    """Basic smoke test for diagonal FIM calculation."""
    params = simple_params
    model = simple_linear_model
    dataset = simple_dataset
    ll_func = simple_gaussian_nll

    fim_diag = calculate_empirical_fim_diagonal(ll_func, params, model, dataset)

    # Expected shape: number of parameters (2 for the linear layer's weight)
    num_params = sum(p.numel() for p in params.values())
    assert fim_diag.shape == (num_params,)
    assert fim_diag.numel() == 2
    assert torch.all(fim_diag >= 0) # FIM diagonal should be non-negative

def test_calculate_fim_diagonal_correctness(simple_params, simple_linear_model, simple_dataset):
    """Test diagonal FIM calculation against analytical result."""
    params = simple_params
    model = simple_linear_model
    dataset = simple_dataset
    ll_func = simple_gaussian_nll

    fim_diag = calculate_empirical_fim_diagonal(ll_func, params, model, dataset)

    # Analytical calculation for the simple linear model and dataset:
    # grads^2 for sample 1: [4.0, 4.0]
    # grads^2 for sample 2: [1.0, 0.0]
    # grads^2 for sample 3: [0.0, 6.25]
    # Sum grads^2: [5.0, 10.25]
    # Expected FIM diag = Sum grads^2 / N = [5.0/3, 10.25/3]
    expected_fim_diag = torch.tensor([5.0/3.0, 10.25/3.0])

    assert fim_diag.shape == (2,)
    assert torch.allclose(fim_diag, expected_fim_diag, atol=1e-5)

def test_calculate_inverse_fim_diagonal(simple_params, simple_linear_model, simple_dataset):
    """Test inverse diagonal FIM calculation."""
    params = simple_params
    model = simple_linear_model
    dataset = simple_dataset
    ll_func = simple_gaussian_nll
    damping = 0.1

    inv_fim_diag = calculate_inverse_fim_diagonal(
        ll_func, params, model, dataset, damping=damping
    )

    # Expected from previous test
    expected_fim_diag = torch.tensor([5.0/3.0, 10.25/3.0])
    expected_inv_fim_diag = 1.0 / (expected_fim_diag + damping)

    assert inv_fim_diag.shape == (2,)
    assert torch.allclose(inv_fim_diag, expected_inv_fim_diag, atol=1e-5)
    assert torch.all(inv_fim_diag > 0) # Inverse should be positive

def test_calculate_full_fim_smoke(simple_params, simple_linear_model, simple_dataset):
    """Basic smoke test for full FIM calculation."""
    params = simple_params
    model = simple_linear_model
    dataset = simple_dataset
    ll_func = simple_gaussian_nll

    fim_full = calculate_empirical_fim_full(ll_func, params, model, dataset)

    num_params = sum(p.numel() for p in params.values())
    assert fim_full.shape == (num_params, num_params)
    assert fim_full.numel() == 4 # 2x2 for this model
    # Check positive semi-definiteness (eigenvalues >= 0)
    try:
        eigenvalues = torch.linalg.eigvalsh(fim_full)
        assert torch.all(eigenvalues >= -1e-6) # Allow small numerical error
    except torch.linalg.LinAlgError:
        pytest.fail("Could not compute eigenvalues for FIM check")

def test_calculate_full_fim_correctness(simple_params, simple_linear_model, simple_dataset):
    """Test full FIM calculation against analytical result."""
    params = simple_params
    model = simple_linear_model
    dataset = simple_dataset
    ll_func = simple_gaussian_nll

    fim_full = calculate_empirical_fim_full(ll_func, params, model, dataset)

    # Analytical calculation:
    # grad1 = [-2.0, -2.0]
    # grad2 = [-1.0,  0.0]
    # grad3 = [ 0.0, -2.5]
    # Sum gg^T = g1*g1^T + g2*g2^T + g3*g3^T
    # g1*g1^T = [[4, 4], [4, 4]]
    # g2*g2^T = [[1, 0], [0, 0]]
    # g3*g3^T = [[0, 0], [0, 6.25]]
    # Sum = [[5, 4], [4, 10.25]]
    # Expected FIM = Sum / N = [[5/3, 4/3], [4/3, 10.25/3]]
    expected_fim_full = torch.tensor([
        [5.0/3.0, 4.0/3.0],
        [4.0/3.0, 10.25/3.0]
    ])

    assert fim_full.shape == (2, 2)
    assert torch.allclose(fim_full, expected_fim_full, atol=1e-5)

def test_calculate_inverse_full_fim(simple_params, simple_linear_model, simple_dataset):
    """Test inverse full FIM calculation."""
    params = simple_params
    model = simple_linear_model
    dataset = simple_dataset
    ll_func = simple_gaussian_nll
    damping = 0.1

    inv_fim_full = calculate_inverse_fim_full(
        ll_func, params, model, dataset, damping=damping
    )

    # Expected from previous test
    expected_fim_full = torch.tensor([
        [5.0/3.0, 4.0/3.0],
        [4.0/3.0, 10.25/3.0]
    ])
    damped_fim = expected_fim_full + torch.eye(2) * damping
    expected_inv_fim_full = torch.linalg.inv(damped_fim)

    assert inv_fim_full.shape == (2, 2)
    assert torch.allclose(inv_fim_full, expected_inv_fim_full, atol=1e-5)

@pytest.mark.parametrize("use_batching", [False, True])
def test_fim_batching(simple_params, simple_linear_model, simple_dataset, use_batching):
    """Test FIM calculations with and without vmap batching."""
    params = simple_params
    model = simple_linear_model
    dataset = simple_dataset
    ll_func = simple_gaussian_nll
    damping = 1e-4
    batch_size = len(dataset) // 2 if use_batching else None

    # Test Diagonal FIM
    fim_diag = calculate_empirical_fim_diagonal(
        ll_func, params, model, dataset, vmap_batch_size=batch_size
    )
    fim_diag_nobatch = calculate_empirical_fim_diagonal(ll_func, params, model, dataset)
    assert torch.allclose(fim_diag, fim_diag_nobatch, atol=1e-6)

    # Test Full FIM
    fim_full = calculate_empirical_fim_full(
        ll_func, params, model, dataset, vmap_batch_size=batch_size
    )
    fim_full_nobatch = calculate_empirical_fim_full(ll_func, params, model, dataset)
    # Use relative tolerance for full FIM comparison due to potential accumulation differences
    assert torch.allclose(fim_full, fim_full_nobatch, rtol=1e-4, atol=1e-6)

    # Test Inverse Diagonal FIM
    inv_fim_diag = calculate_inverse_fim_diagonal(
        ll_func, params, model, dataset, damping=damping, vmap_batch_size=batch_size
    )
    inv_fim_diag_nobatch = calculate_inverse_fim_diagonal(
        ll_func, params, model, dataset, damping=damping
    )
    assert torch.allclose(inv_fim_diag, inv_fim_diag_nobatch, atol=1e-6)

    # Test Inverse Full FIM
    inv_fim_full = calculate_inverse_fim_full(
        ll_func, params, model, dataset, damping=damping, vmap_batch_size=batch_size
    )
    inv_fim_full_nobatch = calculate_inverse_fim_full(
        ll_func, params, model, dataset, damping=damping
    )
    assert torch.allclose(inv_fim_full, inv_fim_full_nobatch, rtol=1e-3, atol=1e-6)

def test_fim_empty_dataset(simple_params, simple_linear_model):
    """Test FIM calculations with an empty dataset."""
    params = simple_params
    model = simple_linear_model
    empty_dataset = []
    ll_func = simple_gaussian_nll
    num_params = sum(p.numel() for p in params.values())

    # Diagonal FIM
    fim_diag = calculate_empirical_fim_diagonal(ll_func, params, model, empty_dataset)
    assert torch.all(fim_diag == 0)
    assert fim_diag.shape == (num_params,)

    # Inverse Diagonal FIM (expect 1/damping)
    damping=0.1
    inv_fim_diag = calculate_inverse_fim_diagonal(ll_func, params, model, empty_dataset, damping=damping)
    assert torch.allclose(inv_fim_diag, torch.full_like(fim_diag, 1.0/damping))

    # Full FIM
    fim_full = calculate_empirical_fim_full(ll_func, params, model, empty_dataset)
    assert torch.all(fim_full == 0)
    assert fim_full.shape == (num_params, num_params)

    # Inverse Full FIM (expect I/damping)
    inv_fim_full = calculate_inverse_fim_full(ll_func, params, model, empty_dataset, damping=damping)
    expected_inv = torch.eye(num_params, device=inv_fim_full.device) / damping
    assert torch.allclose(inv_fim_full, expected_inv)

# Test device handling only if CUDA is available, otherwise skip
if torch.cuda.is_available():
    DEVICES = [torch.device("cpu"), torch.device("cuda")]
else:
    DEVICES = [torch.device("cpu")]

@pytest.mark.parametrize("device", DEVICES)
def test_fim_device_handling(simple_params, simple_linear_model, simple_dataset, device):
    """Test FIM calculations respect the device argument."""
    # Move model and params explicitly (though function should handle it)
    model = simple_linear_model.to(device)
    params = {k: v.to(device) for k,v in simple_params.items()}
    # Dataset items remain on CPU, function should move them
    dataset = simple_dataset
    ll_func = simple_gaussian_nll
    damping = 1e-4
    num_params = sum(p.numel() for p in params.values())

    # Test Diagonal FIM
    fim_diag = calculate_empirical_fim_diagonal(
        ll_func, params, model, dataset, device=device
    )
    assert fim_diag.device == device
    assert fim_diag.shape == (num_params,)

    # Test Full FIM
    fim_full = calculate_empirical_fim_full(
        ll_func, params, model, dataset, device=device
    )
    assert fim_full.device == device
    assert fim_full.shape == (num_params, num_params)

    # Test Inverse Diagonal FIM
    inv_fim_diag = calculate_inverse_fim_diagonal(
        ll_func, params, model, dataset, damping=damping, device=device
    )
    assert inv_fim_diag.device == device
    assert inv_fim_diag.shape == (num_params,)

    # Test Inverse Full FIM
    inv_fim_full = calculate_inverse_fim_full(
        ll_func, params, model, dataset, damping=damping, device=device
    )
    assert inv_fim_full.device == device
    assert inv_fim_full.shape == (num_params, num_params)

# TODO: Add more specific tests for FIM calculation functions
# - Test device handling (CPU vs CUDA if available)
# - Test vmap batching vs no batching
# - Test damping (for inverse functions) - basic covered, more edge cases?
# - Test edge cases (empty dataset)
# - Test full FIM calculation and inversion

# TODO: Add tests for FIM calculation functions (requires more setup)
# - Mock model, params, dataset, log_likelihood
# - Test correctness (maybe analytical for small cases)
# - Test device handling
# - Test vmap batching vs no batching
# - Test damping
# - Test edge cases (empty dataset) 