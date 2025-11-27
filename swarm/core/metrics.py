import numpy as np
from .state import SwarmState


def coverage_extent(state: SwarmState) -> float:
    """
    Rough coverage proxy: area of bounding box around all agents.
    """
    positions = np.array([a.pos for a in state.agents.values()])
    if len(positions) == 0:
        return 0.0
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    return float(np.prod(maxs - mins))


def mean_pairwise_distance(state: SwarmState) -> float:
    """
    Cohesion proxy: average pairwise distance between agents.
    """
    positions = [a.pos for a in state.agents.values()]
    if len(positions) < 2:
        return 0.0
    positions = np.array(positions)
    dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    # exclude self distances (zero diagonal)
    n = len(positions)
    return float(dists.sum() / (n * (n - 1)))


def collision_count(state: SwarmState, threshold: float = 0.1) -> int:
    """
    Count number of agent pairs closer than threshold.
    """
    positions = [a.pos for a in state.agents.values()]
    if len(positions) < 2:
        return 0
    positions = np.array(positions)
    dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    collisions = (dists < threshold).astype(int)
    # zero diagonal and double counted pairs
    collisions = np.triu(collisions, k=1)
    return int(collisions.sum())
