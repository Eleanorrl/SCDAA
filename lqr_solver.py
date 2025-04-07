# lqr_solver.py
import torch # type: ignore
import numpy as np # type: ignore
from scipy.integrate import solve_ivp # type: ignore

class LQRProblem:
    def __init__(self, H, M, sigma, C, D, R, T, time_grid):
        """
        Initializes the LQR problem data and time grid.

        We solve the ODE system for:
          S'(t) = S(t) M D^{-1} M^T S(t) - H^T S(t) - S(t) H - C,   S(T)=R,
          b'(t) = -trace( sigma sigma^T S(t) ),                    b(T)=0,
        so that the value function is  v(t,x) = x^T S(t) x + b(t).

        Parameters
        ----------
        H, M, sigma, C, D, R : torch.Tensor
            Matrices for the LQR problem (2D in typical examples).
        T : float
            Terminal time.
        time_grid : 1D iterable
            Discrete times from 0 to T (does not need to be uniform).
        """
        # Store data as CPU float Tensors
        self.H = H.clone().cpu().float()
        self.M = M.clone().cpu().float()
        self.sigma = sigma.clone().cpu().float()
        self.C = C.clone().cpu().float()
        self.D = D.clone().cpu().float()
        self.R = R.clone().cpu().float()
        self.T = float(T)

        # Convert time_grid to a sorted numpy array
        time_grid_np = np.sort(np.array(time_grid, dtype=np.float64))
        self.time_grid = time_grid_np
        # We also store shape info
        self.dim = self.H.shape[0]

        # Precompute D^-1
        # We'll do some ops in numpy, so let's get a numpy copy
        self.D_inv_np = torch.inverse(self.D).numpy()

        # We'll keep solutions here after we solve the Riccati ODE
        self.S_solutions = None
        self.b_solutions = None
        self.solved_time_grid = None

        # Solve the Riccati equation once at initialization
        self._solve_riccati()

    def _riccati_ode(self, t, y):
        """
        Right-hand side of the ODE system for S(t) and b(t).

        We pack S(t) as a flattened (dim*dim) vector, then b(t) as the last element,
        so y = [S_flat, b].
        """
        dim = self.dim

        # Unpack
        S_flat = y[:dim*dim]
        b_val  = y[-1]

        # Reshape S
        S_mat = S_flat.reshape(dim, dim)

        # S'(t) = S M D^{-1} M^T S - H^T S - S H - C
        # (We assume H, S, C are all stored as numpy arrays.)
        H_np = self.H.numpy()
        M_np = self.M.numpy()
        C_np = self.C.numpy()

        SM = S_mat @ M_np @ self.D_inv_np @ M_np.T @ S_mat
        HS = (H_np.T @ S_mat) + (S_mat @ H_np)
        dSdt = SM - HS - C_np

        # b'(t) = - trace( sigma sigma^T S(t) )
        sigma_np = self.sigma.numpy()
        sigma_sigmaT = sigma_np @ sigma_np.T
        dbdt = - float(np.trace(sigma_sigmaT @ S_mat))

        # Flatten dSdt
        dSdt_flat = dSdt.reshape(-1)

        return np.concatenate([dSdt_flat, np.array([dbdt])])

    def _solve_riccati(self):
        """
        Solve the extended ODE system (for S(t) and b(t)) backward from T to 0.
        We store the solutions at the times in self.time_grid.
        """
        dim = self.dim
        # Final conditions at t = T:
        #   S(T) = R,   b(T) = 0
        R_np = self.R.numpy()
        S_final = R_np.reshape(-1)
        b_final = 0.0
        y_final = np.concatenate([S_final, np.array([b_final])])

        # We'll integrate from t=T down to t=0, sampling at reversed time_grid
        t_eval_desc = self.time_grid[::-1]  # reversed grid

        # scipys ODE system
        def odesys(t, y):
            return self._riccati_ode(t, y)

        sol = solve_ivp(
            fun=odesys,
            t_span=(self.T, 0.0),
            y0=y_final,
            t_eval=t_eval_desc,
            method='RK45',
            rtol=1e-8,
            atol=1e-8
        )

        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")

        # The output is in descending order.
        # We'll flip to ascending order for convenience
        y_desc = sol.y  # shape: (dim*dim + 1, len(t_eval_desc))
        t_desc = sol.t  # shape: (len(t_eval_desc),)

        t_asc = t_desc[::-1].copy()  # must copy() to avoid negative strides
        y_asc = y_desc[:, ::-1].copy()

        # We'll store them as lists of Tensors, then stack
        S_list = []
        b_list = []

        for i in range(len(t_asc)):
            flat = y_asc[:dim*dim, i]
            bval = y_asc[-1, i]
            S_mat = flat.reshape(dim, dim)
            # Convert to torch
            S_list.append(torch.from_numpy(S_mat).float())
            b_list.append(torch.tensor(bval, dtype=torch.float32))

        self.S_solutions = torch.stack(S_list, dim=0)  # shape (len(t_asc), dim, dim)
        self.b_solutions = torch.stack(b_list, dim=0)  # shape (len(t_asc),)
        self.solved_time_grid = torch.from_numpy(t_asc).float()  # shape (len(t_asc),)

    def _find_time_index(self, t):
        """
        Find index in self.solved_time_grid for largest grid time <= t.
        If t < 0, return index=0; if t >= T, return the last index.
        """
        if t <= 0.0:
            return 0
        if t >= self.T:
            return len(self.solved_time_grid) - 1
        idx = torch.searchsorted(self.solved_time_grid, torch.tensor(t), right=True)
        i = int(idx.item()) - 1
        if i < 0: 
            i = 0
        elif i >= len(self.solved_time_grid) - 1:
            i = len(self.solved_time_grid) - 1
        return i

    def value(self, t_tensor, x_tensor):
        """
        Evaluate the value function v(t, x) = x^T S(t) x + b(t)
        for each row of t_tensor, x_tensor.

        t_tensor: shape (N,)
        x_tensor: shape (N, dim)
        returns:  shape (N,)  (the scalar values)
        """
        t_tensor = t_tensor.cpu().float()
        x_tensor = x_tensor.cpu().float()

        values = []
        for ti, xi in zip(t_tensor, x_tensor):
            idx = self._find_time_index(ti.item())
            S_mat = self.S_solutions[idx]  # shape (dim, dim)
            b_val = self.b_solutions[idx].item()
            # compute x^T S x
            val_xSx = xi @ (S_mat @ xi)
            val = val_xSx.item() + b_val
            values.append(val)

        return torch.tensor(values, dtype=torch.float32)

    def optimal_control(self, t_tensor, x_tensor):
        """
        Return the strict optimal control a*(t,x) = -D^{-1} M^T S(t) x for each row of
        t_tensor, x_tensor.

        t_tensor: shape (N,)
        x_tensor: shape (N, dim)
        returns:  shape (N, m)
        """
        t_tensor = t_tensor.cpu().float()
        x_tensor = x_tensor.cpu().float()

        D_inv = torch.inverse(self.D)      # shape (m, m)
        M_T   = self.M.transpose(0, 1)     # shape (m, d)

        controls = []
        for ti, xi in zip(t_tensor, x_tensor):
            idx = self._find_time_index(ti.item())
            S_mat = self.S_solutions[idx]  # shape (d, d)

            Sx = S_mat @ xi               # shape (d,)
            Msx = M_T @ Sx               # shape (m,)
            ctrl = - D_inv @ Msx
            controls.append(ctrl)

        return torch.stack(controls, dim=0)



