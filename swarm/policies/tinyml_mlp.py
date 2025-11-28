import numpy as np
from .base import Policy


class TinyMLPPolicy(Policy):
    def __init__(self, weights=None, hidden=16, k_max=5, seed=0, dim=2):
        self.hidden = hidden
        self.k_max = k_max
        self.dim = dim
        if weights is None:
            self.weights = self.init_weights(dim=dim, k_max=k_max, hidden=hidden, seed=seed)
        else:
            self.weights = weights  # dict of small np arrays, or torch-lite

    @staticmethod
    def obs_dim(dim=2, k_max=5) -> int:
        # pos + vel + neighbors pos/vel + nearest target + resource + boundary distances + battery/role/team
        return dim * (2 * k_max + 6) + 3

    @classmethod
    def init_weights(cls, dim=2, k_max=5, hidden=16, seed=0):
        rng = np.random.default_rng(seed=seed)
        input_dim = cls.obs_dim(dim=dim, k_max=k_max)
        W1 = rng.standard_normal((hidden, input_dim)) * 0.1
        b1 = np.zeros(hidden)
        W2 = rng.standard_normal((dim, hidden)) * 0.1
        b2 = np.zeros(dim)
        return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    def build_observation(self, self_state, neighbor_msgs, visible_targets=None, visible_resource=None):
        # Encode K nearest neighbors into fixed-size feature
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

        # nearest active target relative position
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

        if hasattr(self, "bounds") and self.bounds and len(self.bounds) >= 4:
            xmin, xmax, ymin, ymax = self.bounds
            boundary_feat = np.array(
                [
                    self_state.pos[0] - xmin,
                    xmax - self_state.pos[0],
                    self_state.pos[1] - ymin,
                    ymax - self_state.pos[1],
                ],
                dtype=float,
            )
        else:
            boundary_feat = np.zeros(2 * self.dim, dtype=float)

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
        return feat

    def act(self, obs, return_log_prob: bool = False):
        # obs is np.ndarray
        x = obs
        W1, b1 = self.weights["W1"], self.weights["b1"]
        W2, b2 = self.weights["W2"], self.weights["b2"]
        h = np.tanh(W1 @ x + b1)
        dv = W2 @ h + b2
        return (dv, None) if return_log_prob else dv
