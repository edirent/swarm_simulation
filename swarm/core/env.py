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
    def __init__(self, bounds, obstacles=None, targets=None, tasks=None, sense_radius: float | None = None, enemies=None):
        self.bounds = bounds  # [xmin, xmax, ymin, ymax]
        self.obstacles = obstacles or []
        self.targets = targets or []
        self.tasks = tasks or []
        self.sense_radius = sense_radius
        self.enemy_spawns = enemies or []

    def visible_neighbors(self, self_state, swarm_state):
        """
        Return neighbor AgentState list within sense_radius (if set), otherwise all.
        """
        if self.sense_radius is None:
            return [st for j, st in swarm_state.agents.items() if j != self_state.id]
        vis = []
        for j, st in swarm_state.agents.items():
            if j == self_state.id:
                continue
            d = np.linalg.norm(st.pos - self_state.pos)
            if d <= self.sense_radius:
                vis.append(st)
        return vis

    def visible_targets(self, self_state):
        """
        Return active targets within sense_radius (if set), otherwise all active.
        """
        active = [t for t in self.targets if getattr(t, "active", True)]
        if self.sense_radius is None:
            return active
        vis = []
        for t in active:
            d = np.linalg.norm(t.center - self_state.pos)
            if d <= self.sense_radius:
                vis.append(t)
        return vis

    def enforce_constraints(self, swarm_state, tol=1e-6, damp_on_hit: float | None = 1.0):
        """
        Reflect or damp velocity when hitting bounds, and report boundary contacts.
        Returns:
          boundary_hits: {agent_id: True/False}
        """
        xmin, xmax, ymin, ymax = self.bounds
        boundary_hits = {i: False for i in swarm_state.agents.keys()}
        for i, st in swarm_state.agents.items():
            x, y = st.pos[:2]
            vx, vy = st.vel[:2]

            # X axis
            if x < xmin:
                st.pos[0] = xmin
                st.vel[0] = abs(vx) * (damp_on_hit or 0.0)
                boundary_hits[i] = True
            elif x > xmax:
                st.pos[0] = xmax
                st.vel[0] = -abs(vx) * (damp_on_hit or 0.0)
                boundary_hits[i] = True

            # Y axis
            if y < ymin:
                st.pos[1] = ymin
                st.vel[1] = abs(vy) * (damp_on_hit or 0.0)
                boundary_hits[i] = True
            elif y > ymax:
                st.pos[1] = ymax
                st.vel[1] = -abs(vy) * (damp_on_hit or 0.0)
                boundary_hits[i] = True

            if not boundary_hits[i]:
                boundary_hits[i] = (
                    abs(st.pos[0] - xmin) <= tol
                    or abs(st.pos[0] - xmax) <= tol
                    or abs(st.pos[1] - ymin) <= tol
                    or abs(st.pos[1] - ymax) <= tol
                )
        return boundary_hits

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
