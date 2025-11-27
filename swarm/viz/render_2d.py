import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from ..core.state import SwarmState


class SwarmRenderer2D:
    def __init__(self, bounds, obstacles=None, targets=None):
        self.bounds = bounds
        self.obstacles = obstacles or []
        self.targets = targets or []
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        backend = plt.get_backend().lower()
        self._interactive = backend not in {"agg", "pdf", "svg"}
        if self._interactive:
            plt.ion()
        self.scat = None
        self.obstacle_patches = []
        self.target_patches = []
        self.target_labels = []
        self.ax.set_xlim(bounds[0], bounds[1])
        self.ax.set_ylim(bounds[2], bounds[3])
        self.ax.set_aspect("equal")
        self._draw_static()

    def _draw_static(self):
        for idx, obs in enumerate(self.obstacles):
            patch = mpatches.Circle(obs.center[:2], obs.radius, color="gray", alpha=0.25, zorder=1)
            self.ax.add_patch(patch)
            self.obstacle_patches.append(patch)
            self.ax.text(obs.center[0], obs.center[1], f"O{idx}", ha="center", va="center", fontsize=8, color="black")
        for idx, tgt in enumerate(self.targets):
            patch = mpatches.Circle(tgt.center[:2], tgt.radius, color="green", alpha=0.5, zorder=2, ec="darkgreen")
            self.ax.add_patch(patch)
            self.target_patches.append(patch)
            label = self.ax.text(
                tgt.center[0],
                tgt.center[1],
                f"T{idx}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                weight="bold",
                zorder=3,
            )
            self.target_labels.append(label)

    def _update_targets(self):
        for tgt, patch in zip(self.targets, self.target_patches):
            color = "green" if tgt.active else "red"
            patch.set_color(color)
            patch.set_edgecolor("darkgreen" if tgt.active else "darkred")
            patch.set_alpha(0.5 if tgt.active else 0.25)

    def render(self, swarm_state: SwarmState):
        positions = np.array([a.pos for a in swarm_state.agents.values()])
        if positions.size == 0:
            return
        if self.scat is None:
            self.scat = self.ax.scatter(positions[:, 0], positions[:, 1], c="blue", s=20, zorder=4, alpha=0.8)
        else:
            self.scat.set_offsets(positions[:, :2])
        self._update_targets()
        self.ax.set_title(f"t={swarm_state.t:.2f}")
        if self._interactive:
            plt.pause(0.001)
