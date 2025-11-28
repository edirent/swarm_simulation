import torch
import torch.nn as nn
import numpy as np

from .base import Policy


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TinyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class NNPolicy(Policy):
    def __init__(self, model: nn.Module, k_max: int = 5, dim: int = 2, device: str | None = None, stochastic: bool = False, noise_std: float = 0.1, bounds=None):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.k_max = k_max
        self.dim = dim
        self.stochastic = stochastic
        self.noise_std = noise_std
        self.bounds = bounds
        self.model.eval()

    @staticmethod
    def obs_dim(dim=2, k_max=5) -> int:
        # pos + vel + neighbors pos/vel + nearest target + resource + boundary distances + battery/role/team
        return dim * (2 * k_max + 6) + 3

    def build_observation(self, self_state, neighbor_msgs, visible_targets=None, visible_resource=None):
        k_max = self.k_max
        ps = np.array([m.pos for m in neighbor_msgs]) if neighbor_msgs else np.zeros((0, len(self_state.pos)))
        vs = np.array([m.vel for m in neighbor_msgs]) if neighbor_msgs else np.zeros((0, len(self_state.vel)))

        if len(ps) > k_max:
            ps = ps[:k_max]
            vs = vs[:k_max]
        else:
            pad = k_max - len(ps)
            if pad > 0:
                ps = np.vstack([ps, np.zeros((pad, ps.shape[1]))])
                vs = np.vstack([vs, np.zeros((pad, vs.shape[1]))])

        if visible_targets:
            ds = [np.linalg.norm(t.center - self_state.pos) for t in visible_targets]
            j = int(np.argmin(ds))
            nearest = visible_targets[j].center - self_state.pos
        else:
            nearest = np.zeros_like(self_state.pos)

        if visible_resource is not None:
            res_vec = visible_resource.center - self_state.pos
        else:
            res_vec = np.zeros_like(self_state.pos)

        if self.bounds and len(self.bounds) >= 4:
            xmin, xmax, ymin, ymax = self.bounds
            boundary_feat = np.array(
                [
                    self_state.pos[0] - xmin,
                    xmax - self_state.pos[0],
                    self_state.pos[1] - ymin,
                    ymax - self_state.pos[1],
                ],
                dtype=np.float32,
            )
        else:
            boundary_feat = np.zeros(2 * self.dim, dtype=np.float32)

        feat = np.concatenate([
            self_state.pos,
            self_state.vel,
            ps.flatten(),
            vs.flatten(),
            nearest,
            res_vec,
            boundary_feat,
            np.array([self_state.battery, self_state.role, self_state.team], dtype=float)
        ])
        return feat.astype(np.float32)

    def act(self, obs: np.ndarray, return_log_prob: bool = False):
        with torch.no_grad():
            x = torch.from_numpy(obs).to(self.device)
            mean = self.model(x)
            if self.stochastic and self.noise_std > 0:
                dist = torch.distributions.Normal(mean, self.noise_std)
                sample = dist.sample()
                dv = sample.cpu().numpy()
                log_prob = dist.log_prob(sample).sum().cpu().item()
            else:
                dv = mean.cpu().numpy()
                log_prob = None
        if return_log_prob:
            return dv, log_prob
        return dv

    def log_prob(self, obs: np.ndarray, action: np.ndarray):
        x = torch.from_numpy(obs).to(self.device)
        act = torch.from_numpy(action).to(self.device)
        mean = self.model(x)
        dist = torch.distributions.Normal(mean, self.noise_std if self.noise_std > 0 else 1e-6)
        return dist.log_prob(act).sum()
