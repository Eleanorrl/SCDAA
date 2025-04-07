# ActorOnly
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Assume that SoftLQRProblem is defined (as in Exercise 2) and imported.
# For this example we assume it is available as SoftLQRProblem.
# It provides:
#   - S_solutions and _find_time_index() (from solving the soft Riccati ODE)
#   - The optimal relaxed control is given by:
#         μ*(t,x) = -[D + (τ/(2γ²)) I]⁻¹ Mᵀ S(t) x
#   - Here D, τ and γ are parameters.

# ----------------- Actor Neural Network -----------------
class ActorNN(nn.Module):
    def __init__(self, state_dim, control_dim, hidden_size=256):
        """
        Actor network that takes time t and state x and outputs a control mean.
        Inputs:
            state_dim: dimension of state x.
            control_dim: dimension of control a.
            hidden_size: width of hidden layers.
        """
        super(ActorNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, control_dim)
        )
        
    def forward(self, t, x):
        # t: Tensor of shape (batch, 1), x: Tensor of shape (batch, state_dim)
        inp = torch.cat([t, x], dim=1)
        mean = self.net(inp)
        return mean

# ----------------- Helper Function: Target Mean Calculation -----------------
def compute_target_mean(soft_problem, t, x):
    """
    Given a soft LQR problem (with solved Riccati solution), time t (scalar tensor),
    and state x (tensor of shape (state_dim,)), compute the target optimal mean control:
        μ*(t,x) = -[D + (τ/(2γ²)) I]⁻¹ Mᵀ S(t) x.
    """
    # Find the appropriate index for time t
    idx = soft_problem._find_time_index(t.item())
    S_t = soft_problem.S_solutions[idx]  # S(t)
    m = soft_problem.D.shape[0]           # control dimension
    # Form the modified D matrix: A = D + (τ/(2γ²)) I
    A = soft_problem.D + (soft_problem.tau/(2*soft_problem.gamma**2)) * torch.eye(m)
    A_inv = torch.inverse(A)
    # Compute target mean: -A_inv * Mᵀ * (S(t)x)
    target_mean = - A_inv @ soft_problem.M.transpose(0, 1) @ (S_t @ x)
    return target_mean

# ----------------- Generate Training Data for Actor -----------------
def generate_actor_data(soft_problem, T, num_samples, state_range=(-3, 3)):
    """
    Generate training data for the actor by sampling (t, x) pairs.
    For each sample, t is drawn uniformly from [0, T] and x is drawn from a uniform
    distribution on [state_range[0], state_range[1]]^state_dim.
    Returns tensors:
        ts: (num_samples, 1)
        xs: (num_samples, state_dim)
        targets: (num_samples, control_dim) with target control means.
    """
    ts = []
    xs = []
    targets = []
    state_dim = soft_problem.H.shape[0]
    m = soft_problem.D.shape[0]  # control dimension
    for _ in range(num_samples):
        t_val = np.random.uniform(0, T)
        x_val = np.random.uniform(state_range[0], state_range[1], size=(state_dim,))
        t_tensor = torch.tensor([t_val], dtype=torch.float32)
        x_tensor = torch.tensor(x_val, dtype=torch.float32)
        target = compute_target_mean(soft_problem, t_tensor, x_tensor)
        ts.append(t_tensor)
        xs.append(x_tensor)
        targets.append(target)
    ts = torch.stack(ts)       # shape: (num_samples, 1)
    xs = torch.stack(xs)       # shape: (num_samples, state_dim)
    targets = torch.stack(targets)  # shape: (num_samples, control_dim)
    return ts, xs, targets

