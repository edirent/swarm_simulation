import matplotlib.pyplot as plt
import numpy as np
from ..core.state import SwarmState


class SwarmRenderer2D:
    def __init__(self, bounds):
        self.bounds = bounds
        self.fig, self.ax = plt.subplots()
        backend = plt.get_backend().lower()
        self._interactive = backend not in {"agg", "pdf", "svg"}
        if self._interactive:
            plt.ion()
        self.scat = None
        self.ax.set_xlim(bounds[0], bounds[1])
        self.ax.set_ylim(bounds[2], bounds[3])
        self.ax.set_aspect("equal")

    def render(self, swarm_state: SwarmState):
        positions = np.array([a.pos for a in swarm_state.agents.values()])
        if positions.size == 0:
            return
        if self.scat is None:
            self.scat = self.ax.scatter(positions[:, 0], positions[:, 1], c="blue")
        else:
            self.scat.set_offsets(positions[:, :2])
        self.ax.set_title(f"t={swarm_state.t:.2f}")
        if self._interactive:
            plt.pause(0.001)
