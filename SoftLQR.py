# SoftLQR

import torch # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from scipy.integrate import solve_ivp # type: ignore
from lqr_solver import LQRProblem

# ----------------- Soft LQR Problem (Exercise 2) -----------------
class SoftLQRProblem(LQRProblem):
    def __init__(self, H, M, sigma, C, D, R, T, time_grid, tau, gamma):
        """
        tau: entropic regularization strength.
        gamma: variance parameter of the reference normal density.
        """
        self.tau = tau
        self.gamma = gamma
        super().__init__(H, M, sigma, C, D, R, T, time_grid)
    
    def _riccati_ode(self, t, y):
        # Override to incorporate entropic regularization.
        dim = self.dim
        S_flat = y[:dim*dim]
        b_val  = y[-1]
        S_mat = S_flat.reshape(dim, dim)
        
        H_np = self.H.numpy()
        M_np = self.M.numpy()
        C_np = self.C.numpy()
        
        # Define A = D + (tau/(2*gamma^2)) I
        m = self.D.shape[0]
        A = self.D.numpy() + (self.tau/(2*self.gamma**2)) * np.eye(m, dtype=np.float64)
        A_inv = np.linalg.inv(A)
        
        SM = S_mat @ M_np @ A_inv @ M_np.T @ S_mat
        HS = (H_np.T @ S_mat) + (S_mat @ H_np)
        dSdt = SM - HS - C_np
        
        # For b'(t), include an additional constant from the entropic term:
        sigma_np = self.sigma.numpy()
        sigma_sigmaT = sigma_np @ sigma_np.T
        detA = np.linalg.det(A)
        # Constant: C_D_tau_gamma = -tau * ln[(tau^(m/2)/gamma^m) * det(A)^(-1/2)]
        C_D_tau_gamma = - self.tau * np.log((self.tau**(m/2))/(self.gamma**m) * (detA)**(-1/2))
        dbdt = - float(np.trace(sigma_sigmaT @ S_mat)) - C_D_tau_gamma
        
        dSdt_flat = dSdt.reshape(-1)
        return np.concatenate([dSdt_flat, np.array([dbdt])])
    
    def soft_optimal_control(self, t_tensor, x_tensor):
        """
        Returns a sample from the optimal soft control distribution.
        The mean is given by:
          - (D + (tau/(2*gamma^2)) I)^{-1} M^T S(t)x,
        and the covariance is:
          tau * (D + (tau/(2*gamma^2)) I).
        """
        t_tensor = t_tensor.cpu().float()
        x_tensor = x_tensor.cpu().float()
        m = self.D.shape[0]
        A = self.D + (self.tau/(2*self.gamma**2)) * torch.eye(m)
        A_inv = torch.inverse(A)
        
        mean_controls = []
        for ti, xi in zip(t_tensor, x_tensor):
            idx = self._find_time_index(ti.item())
            S_mat = self.S_solutions[idx]
            mean_control = - A_inv @ self.M.transpose(0, 1) @ (S_mat @ xi)
            mean_controls.append(mean_control)
        mean_controls = torch.stack(mean_controls, dim=0)
        
        covariance = self.tau * A  # Covariance matrix as per derivation.
        actions = []
        for i in range(mean_controls.shape[0]):
            dist = torch.distributions.MultivariateNormal(mean_controls[i], covariance_matrix=covariance)
            actions.append(dist.sample())
        return torch.stack(actions, dim=0)

# ----------------- Trajectory Simulation -----------------
def simulate_trajectory(problem, x0, T, N, use_soft=False):
    """
    Simulates one trajectory using Euler–Maruyama.
    If use_soft is True, uses soft optimal control; otherwise uses strict control.
    """
    dt = T / N
    t_grid = torch.linspace(0, T, N+1)
    x = x0.clone()
    traj = [x.numpy()]
    for n in range(N):
        t_val = torch.tensor([t_grid[n]])
        if use_soft:
            a = problem.soft_optimal_control(t_val, x.unsqueeze(0))[0]
        else:
            a = problem.optimal_control(t_val, x.unsqueeze(0))[0]
        dW = torch.randn(x.shape) * np.sqrt(dt)
        x = x + dt * (problem.H @ x + problem.M @ a) + problem.sigma @ dW
        traj.append(x.numpy())
    return np.array(traj)

