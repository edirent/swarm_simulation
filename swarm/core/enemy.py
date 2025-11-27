import numpy as np

from .state import AgentState


class Enemy:
    def __init__(self, id: int, pos, speed: float = 1.2, sense: float = 8.0, radius: float = 0.6):
        self.state = AgentState(
            id=id,
            pos=np.array(pos, dtype=float),
            vel=np.zeros(2, dtype=float),
            battery=1.0,
            role=0,
            task_id=None,
            alive=True,
        )
        self.speed = speed
        self.sense = sense
        self.radius = radius

    def step(self, agents: list[AgentState], dt: float, bounds):
        if not self.state.alive or not agents:
            return
        # chase nearest alive agent within sense; else drift
        dists = [(np.linalg.norm(a.pos - self.state.pos), a) for a in agents if a.alive]
        if not dists:
            return
        d_min, target = min(dists, key=lambda x: x[0])
        if d_min <= self.sense:
            direction = target.pos - self.state.pos
            n = np.linalg.norm(direction) + 1e-6
            desired_vel = direction / n * self.speed
        else:
            desired_vel = np.zeros_like(self.state.vel)
        self.state.vel = desired_vel
        self.state.pos = self.state.pos + self.state.vel * dt
        self._clamp(bounds)

    def _clamp(self, bounds):
        xmin, xmax, ymin, ymax = bounds
        x, y = self.state.pos[:2]
        vx, vy = self.state.vel[:2]
        if x < xmin:
            self.state.pos[0] = xmin
            self.state.vel[0] = abs(vx)
        elif x > xmax:
            self.state.pos[0] = xmax
            self.state.vel[0] = -abs(vx)
        if y < ymin:
            self.state.pos[1] = ymin
            self.state.vel[1] = abs(vy)
        elif y > ymax:
            self.state.pos[1] = ymax
            self.state.vel[1] = -abs(vy)
