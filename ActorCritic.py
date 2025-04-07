from lqr_module import SoftLQRProblem, ActorNN, CriticNN
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ---------- Helper Function: Actor Distribution ----------
def get_actor_distribution(actor, t, x, fixed_cov):
    """
    Given the actor network, time t (tensor of shape (batch,1)) and state x (tensor of shape (batch, state_dim)),
    returns a MultivariateNormal distribution with mean predicted by the actor and fixed covariance.
    """
    mean = actor(t, x)  # shape: (batch, control_dim)
    return torch.distributions.MultivariateNormal(mean, covariance_matrix=fixed_cov)

# ---------- Episode Simulation Function ----------
def simulate_episode(actor, critic, soft_problem, T, N, actor_cov):
    dt = T / N
    t_grid = torch.linspace(0, T, N+1)
    state_dim = soft_problem.H.shape[0]
    # Sample initial state uniformly from [-2,2]^state_dim.
    x = torch.tensor(np.random.uniform(-2, 2, size=(state_dim,)), dtype=torch.float32)
    
    times = []
    states = []
    actions = []
    logprobs = []  # Keep these as tensors (do not call .item())
    costs = []
    critic_values = []
    
    for n in range(N):
        t = t_grid[n].unsqueeze(0)  # shape: (1,)
        times.append(t.item())
        states.append(x.clone())
        
        # Compute critic estimate V(t,x) = xᵀ S(t)x + b(t)
        # Ensure x is shaped as (1, state_dim, 1)
        x_exp = x.unsqueeze(0).unsqueeze(2)  # shape: (1, state_dim, 1)
        S, b = critic(t.unsqueeze(0))         # critic output for batch size 1.
        v_current = torch.bmm(x_exp.transpose(1,2), torch.bmm(S, x_exp)).squeeze() + b.squeeze()
        # Detach critic values for use as constants in the actor loss.
        critic_values.append(v_current.detach().item())
        
        # Sample action using actor.
        dist = get_actor_distribution(actor, t.unsqueeze(0), x.unsqueeze(0), actor_cov)
        a = dist.sample().squeeze(0)  # Now a has shape (control_dim,)
        actions.append(a.clone())
        # Append the log probability tensor (do not call .item() here).
        logprobs.append(dist.log_prob(a.unsqueeze(0)))
        
        # Running cost: f = xᵀ C x + aᵀ D a.
        cost = (x @ (soft_problem.C @ x) + a @ (soft_problem.D @ a)).item()
        costs.append(cost)
        
        # Euler–Maruyama state update.
        dW = torch.randn(x.shape) * np.sqrt(dt)
        x = x + dt * (soft_problem.H @ x + soft_problem.M @ a) + soft_problem.sigma @ dW
        
    # Terminal step.
    t_final = t_grid[-1].unsqueeze(0)
    times.append(t_final.item())
    states.append(x.clone())
    x_exp = x.unsqueeze(0).unsqueeze(2)  # shape: (1, state_dim, 1)
    S, b = critic(t_final.unsqueeze(0))
    v_final = torch.bmm(x_exp.transpose(1,2), torch.bmm(S, x_exp)).squeeze() + b.squeeze()
    critic_values.append(v_final.detach().item())
    terminal_cost = (x @ (soft_problem.R @ x)).item()
    
    # Compute TD errors: δ_n = V(t_{n+1}, x_{n+1}) - V(t_n, x_n)
    td_errors = []
    for n in range(N):
        delta = critic_values[n+1] - critic_values[n]
        td_errors.append(delta)
    
    episode_data = {
        'times': times,
        'states': states,
        'actions': actions,
        # Stack the logprob tensors into one tensor.
        'logprobs': torch.cat(logprobs, dim=0),  
        'costs': costs,
        'td_errors': td_errors,
        'terminal_cost': terminal_cost,
        'dt': dt
    }
    return episode_data

# ---------- Main Actor-Critic Training Loop (Exercise 5) ----------
if __name__ == "__main__":
    # Define problem parameters.
    T = 0.5
    state_dim = 2
    control_dim = 2
    H = torch.tensor([[1.0, 0.8],
                      [0.1, 0.3]], dtype=torch.float32)
    M = torch.tensor([[0.3, 0.1],
                      [0.8, 1.0]], dtype=torch.float32)
    sigma = 0.5 * torch.eye(2)
    C = 10 * torch.eye(2)
    # For actor–critic, set D = I.
    D = torch.eye(2)
    R = 100 * torch.eye(2)
    time_grid = np.linspace(0, T, 1000)
    tau = 0.5
    gamma = 1.0
    
    # Create soft LQR problem instance.
    soft_problem = SoftLQRProblem(H, M, sigma, C, D, R, T, time_grid, tau, gamma)
    
    # Create actor and critic networks.
    actor = ActorNN(state_dim, control_dim, hidden_size=256)
    critic = CriticNN(hidden_size=512, device=torch.device("cpu"))
    
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-10)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-10)
    
    # Fixed covariance for actor policy: use theoretical covariance.
    m = D.shape[0]
    A = D + (tau/(2*gamma**2)) * torch.eye(m)
    actor_cov = tau * A  # fixed covariance.
    
    num_episodes = 5000
    actor_losses = []
    critic_losses = []
    
    for episode in range(num_episodes):
        ep_data = simulate_episode(actor, critic, soft_problem, T, N=100, actor_cov=actor_cov)
        # td_errors is a list of Python floats.
        td_errors = torch.tensor(ep_data['td_errors'], dtype=torch.float32)
        # logprobs is now a tensor from stacking the raw log probabilities.
        logprobs = ep_data['logprobs']
        costs = torch.tensor(ep_data['costs'], dtype=torch.float32)
        dt_ep = ep_data['dt']
        
        extra_terms = costs + tau * logprobs
        # Compute actor loss using REINFORCE-style update:
        # We treat td_errors as constant (detached) and backpropagate through logprobs.
        actor_loss = - (logprobs.dot(td_errors) + extra_terms.sum() * dt_ep)
        
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        actor_losses.append(actor_loss.item())
        
        # Critic update: target return G = (∑ f*dt) + terminal cost.
        G = dt_ep * costs.sum() + ep_data['terminal_cost']
        x0 = ep_data['states'][0].unsqueeze(0)
        t0 = torch.tensor([[0.0]], dtype=torch.float32)
        x0_exp = x0.unsqueeze(2)  # shape: (1, state_dim, 1)
        S, b = critic(t0)
        v_pred = torch.bmm(x0_exp.transpose(1,2), torch.bmm(S, x0_exp)).squeeze() + b.squeeze()
        critic_loss = (v_pred - G)**2
        
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        critic_losses.append(critic_loss.item())
        
        if episode % 500 == 0:
            print(f"Episode {episode}: Actor loss = {actor_loss.item():.2e}, Critic loss = {critic_loss.item():.2e}")
    
    plt.figure()
    plt.plot(actor_losses)
    plt.xlabel("Episode")
    plt.ylabel("Actor Loss")
    plt.title("Actor Loss during Training")
    plt.yscale("log")
    plt.show()
    
    plt.figure()
    plt.plot(critic_losses)
    plt.xlabel("Episode")
    plt.ylabel("Critic Loss")
    plt.title("Critic Loss during Training")
    plt.yscale("log")
    plt.show()
