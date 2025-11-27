import numpy as np
from .base import Policy


class BoidsPolicy(Policy):
    def __init__(self, w_sep=1.0, w_align=0.5, w_coh=0.5, max_speed=1.0):
        self.w_sep = w_sep
        self.w_align = w_align
        self.w_coh = w_coh
        self.max_speed = max_speed

    def build_observation(self, self_state, neighbor_msgs, visible_targets=None):
        return (self_state, neighbor_msgs)

    def act(self, obs, return_log_prob: bool = False):
        self_state, neighbor_msgs = obs
        if not neighbor_msgs:
            return (np.zeros_like(self_state.vel), None) if return_log_prob else np.zeros_like(self_state.vel)

        ps = np.array([m.pos for m in neighbor_msgs])
        vs = np.array([m.vel for m in neighbor_msgs])

        p = self_state.pos
        v = self_state.vel

        # simple boids-like terms
        center = ps.mean(axis=0)
        coh = center - p

        sep = (p - ps)
        dist = np.linalg.norm(sep, axis=1, keepdims=True) + 1e-6
        sep = (sep / dist**2).sum(axis=0)

        align = vs.mean(axis=0) - v

        dv = self.w_sep * sep + self.w_align * align + self.w_coh * coh
        # clamp
        if np.linalg.norm(dv) > self.max_speed:
            dv = dv / np.linalg.norm(dv) * self.max_speed
        return (dv, None) if return_log_prob else dv
