import numpy as np
from .base import Policy


class BoidsPolicy(Policy):
    def __init__(self, w_sep=1.0, w_align=0.5, w_coh=0.5, w_target=0.3, w_resource=0.8, max_speed=1.0):
        self.w_sep = w_sep
        self.w_align = w_align
        self.w_coh = w_coh
        self.w_target = w_target
        self.w_resource = w_resource
        self.max_speed = max_speed

    def build_observation(self, self_state, neighbor_msgs, visible_targets=None, visible_resource=None):
        return (self_state, neighbor_msgs, visible_targets, visible_resource)

    def act(self, obs, return_log_prob: bool = False):
        self_state, neighbor_msgs, visible_targets, visible_resource = obs
        if not neighbor_msgs:
            base = np.zeros_like(self_state.vel)
        else:
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

            base = self.w_sep * sep + self.w_align * align + self.w_coh * coh

        target_term = np.zeros_like(self_state.vel)
        if visible_targets:
            ds = [np.linalg.norm(t.center - self_state.pos) for t in visible_targets]
            j = int(np.argmin(ds))
            vec = visible_targets[j].center - self_state.pos
            n = np.linalg.norm(vec) + 1e-6
            target_term = vec / n

        resource_term = np.zeros_like(self_state.vel)
        if visible_resource is not None:
            vec_r = visible_resource.center - self_state.pos
            n = np.linalg.norm(vec_r) + 1e-6
            resource_term = vec_r / n

        dv = base + self.w_target * target_term + self.w_resource * resource_term
        # clamp
        if np.linalg.norm(dv) > self.max_speed:
            dv = dv / np.linalg.norm(dv) * self.max_speed
        return (dv, None) if return_log_prob else dv
