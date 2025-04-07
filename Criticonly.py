# Criticonly
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- Assume SoftLQRProblem is already implemented and imported ---
# For example, SoftLQRProblem is defined as in Exercise 2 and provides:
#   - soft_optimal_control(t_tensor, x_tensor)
#   - (other methods to solve the Riccati ODE, etc.)
# We also assume that D is set to the identity.

# ----------------- Critic Neural Network -----------------
class CriticNN(nn.Module):
    def __init__(self, hidden_size=512, device=torch.device("cpu")):
        super(CriticNN, self).__init__()
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        # Output 4 numbers that will form a 2x2 matrix
        self.matrix_layer = nn.Linear(hidden_size, 4)
        self.offset_layer = nn.Linear(hidden_size, 1)
        
    def forward(self, t):
        # t: shape (batch_size, 1)
        h = self.fc(t)
        mat = self.matrix_layer(h)  # shape: (batch_size, 4)
        mat = mat.view(-1, 2, 2)
        # Force positive semidefiniteness: S = L Lᵀ + εI
        S = torch.bmm(mat, mat.transpose(1,2)) + 1e-3 * torch.eye(2, device=self.device).unsqueeze(0)
        b = self.offset_layer(h)
        return S, b

def critic_value(critic, t, x):
    """
    Given critic network outputs S(t) and b(t),
    computes V(t,x)= xᵀ S(t)x + b(t).
    t: Tensor (batch_size, 1)
    x: Tensor (batch_size, 2)
    """
    S, b = critic(t)
    x_expanded = x.unsqueeze(2)  # (batch_size, 2, 1)
    quad = torch.bmm(x_expanded.transpose(1,2), torch.bmm(S, x_expanded)).squeeze()
    return quad + b.squeeze()

# ----------------- Simulation of Cost-to-Go -----------------
def simulate_cost(soft_problem, t0, x0, T, N):
    """
    Simulate a trajectory from time t0 with initial state x0 using the soft optimal control.
    Compute the total cost-to-go:
      J = ∫ₜᵀ [xᵀ C x + aᵀ D a] ds + x_Tᵀ R x_T.
    
    Parameters:
      t0: starting time (float)
      x0: starting state (torch.Tensor of shape (2,))
      T: terminal time
      N: number of time steps
    
    Returns:
      cost (float)
    """
    dt = (T - t0) / N
    t = t0
    x = x0.clone()
    total_cost = 0.0
    for _ in range(N):
        t_tensor = torch.tensor([t], dtype=torch.float32)
        # Sample action using the optimal soft control
        a = soft_problem.soft_optimal_control(t_tensor, x.unsqueeze(0))[0]
        # Running cost: xᵀ C x + aᵀ D a (here D = I)
        running_cost = (x @ (soft_problem.C @ x) + a @ (soft_problem.D @ a)) * dt
        total_cost += running_cost.item()
        # Euler–Maruyama update for state
        dW = torch.randn(x.shape) * np.sqrt(dt)
        x = x + dt * (soft_problem.H @ x + soft_problem.M @ a) + soft_problem.sigma @ dW
        t += dt
    # Terminal cost: xᵀ R x
    total_cost += (x @ (soft_problem.R @ x)).item()
    return total_cost

# ----------------- Generate Training Data for Critic -----------------
def generate_training_data(soft_problem, T, num_samples, N):
    """
    Generate training data by sampling starting times and states,
    then simulating the cost-to-go from (t, x).
    
    Returns:
      ts: Tensor of shape (num_samples, 1)
      xs: Tensor of shape (num_samples, 2)
      targets: Tensor of shape (num_samples,)
    """
    ts, xs, targets = [], [], []
    for _ in range(num_samples):
        t0 = np.random.uniform(0, T)
        x0 = torch.tensor(np.random.uniform(-3, 3, size=(2,)), dtype=torch.float32)
        cost = simulate_cost(soft_problem, t0, x0, T, N)
        ts.append([t0])
        xs.append(x0)
        targets.append(cost)
    ts = torch.tensor(ts, dtype=torch.float32)
    xs = torch.stack(xs)
    targets = torch.tensor(targets, dtype=torch.float32)
    return ts, xs, targets

