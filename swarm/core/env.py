import numpy as np


class Obstacle:
    def __init__(self, center, radius):
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)


class Target:
    def __init__(self, center, radius=0.3):
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        self.active = True


class SwarmEnv:
    def __init__(self, bounds, obstacles=None, targets=None, tasks=None):
        self.bounds = bounds  # [xmin, xmax, ymin, ymax]
        self.obstacles = obstacles or []
        self.targets = targets or []
        self.tasks = tasks or []

    def enforce_constraints(self, swarm_state):
        """
        Clamp agent positions inside bounds.
        """
        xmin, xmax, ymin, ymax = self.bounds
        for st in swarm_state.agents.values():
            x, y = st.pos[:2]
            st.pos[0] = min(max(x, xmin), xmax)
            st.pos[1] = min(max(y, ymin), ymax)

    def check_collisions(self, swarm_state):
        """
        Returns:
          collisions: {agent_id: True/False}
          hits: {agent_id: [target_idx, ...]}
        """
        collisions = {i: False for i in swarm_state.agents.keys()}
        hits = {i: [] for i in swarm_state.agents.keys()}

        for i, st in swarm_state.agents.items():
            p = st.pos

            for obs in self.obstacles:
                d = np.linalg.norm(p - obs.center)
                if d <= obs.radius:
                    collisions[i] = True
                    break

            for idx, tgt in enumerate(self.targets):
                if not tgt.active:
                    continue
                d = np.linalg.norm(p - tgt.center)
                if d <= tgt.radius:
                    hits[i].append(idx)

        for idx, tgt in enumerate(self.targets):
            for _, lst in hits.items():
                if idx in lst:
                    tgt.active = False

        return collisions, hits

    def all_targets_done(self):
        return bool(self.targets) and all(not t.active for t in self.targets)
