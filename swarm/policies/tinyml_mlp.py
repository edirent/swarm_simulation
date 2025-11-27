import numpy as np
from .base import Policy


class TinyMLPPolicy(Policy):
    def __init__(self, weights):
        self.weights = weights  # dict of small np arrays, or torch-lite

    def build_observation(self, self_state, neighbor_msgs, k_max=5):
        # Example: encode K nearest neighbors into fixed-size feature
        ps = np.array([m.pos for m in neighbor_msgs]) if neighbor_msgs else np.zeros((0, len(self_state.pos)))
        vs = np.array([m.vel for m in neighbor_msgs]) if neighbor_msgs else np.zeros((0, len(self_state.vel)))

        # pad or truncate
        if len(ps) > k_max:
            ps = ps[:k_max]
            vs = vs[:k_max]
        else:
            pad = k_max - len(ps)
            if pad > 0:
                ps = np.vstack([ps, np.zeros((pad, ps.shape[1]))])
                vs = np.vstack([vs, np.zeros((pad, vs.shape[1]))])

        feat = np.concatenate([
            self_state.pos,
            self_state.vel,
            ps.flatten(),
            vs.flatten(),
            np.array([self_state.battery, self_state.role], dtype=float)
        ])
        return feat

    def act(self, obs):
        # obs is np.ndarray
        # forward through tiny MLP (hand-coded or via tiny framework)
        x = obs
        W1, b1 = self.weights["W1"], self.weights["b1"]
        W2, b2 = self.weights["W2"], self.weights["b2"]
        h = np.tanh(W1 @ x + b1)
        dv = W2 @ h + b2
        return dv
