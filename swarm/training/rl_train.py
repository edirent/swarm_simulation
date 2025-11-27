"""
Lightweight CTDE-friendly training loop skeleton.
- Parameter sharing: all agents use the same policy instance.
- Continuous actions (delta_v) assumed.
This stays dependency-light (NumPy only) so it runs even without deep learning libs.
Plug in your optimizer/framework via the `update_fn` callback.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

import numpy as np


@dataclass
class StepRecord:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    done: bool
    info: dict


class TrajectoryBuffer:
    def __init__(self):
        self.data: Dict[int, List[StepRecord]] = {}

    def add(self, agent_id: int, obs, action, reward: float, done: bool, info=None):
        rec = StepRecord(obs=obs, action=action, reward=reward, done=done, info=info or {})
        self.data.setdefault(agent_id, []).append(rec)

    def clear(self):
        self.data.clear()

    def compute_returns(self, gamma: float = 0.99) -> Dict[int, List[float]]:
        """
        Compute discounted returns per-agent (episode-length trajectories).
        """
        returns: Dict[int, List[float]] = {}
        for aid, steps in self.data.items():
            G = 0.0
            ret_seq: List[float] = []
            for step in reversed(steps):
                G = step.reward + gamma * G
                ret_seq.append(G)
            ret_seq.reverse()
            returns[aid] = ret_seq
        return returns


def collect_episode(simulator, max_steps: int = 1000, gamma: float = 0.99) -> Tuple[TrajectoryBuffer, Dict[int, List[float]]]:
    """
    Roll out one episode with shared policy (inside simulator agents), gather trajectories.
    """
    buf = TrajectoryBuffer()
    for _ in range(max_steps):
        state, rewards, collisions, done, hits, logs = simulator.step(return_logs=True)
        for aid, log in logs.items():
            buf.add(
                agent_id=aid,
                obs=log["obs"],
                action=log["action"],
                reward=rewards.get(aid, 0.0),
                done=done,
                info={"collided": collisions.get(aid, False), "hits": hits.get(aid, [])},
            )
        if done:
            break
    returns = buf.compute_returns(gamma=gamma)
    return buf, returns


def ctde_training_loop(simulator, update_fn, episodes: int = 10, gamma: float = 0.99):
    """
    Centralized training, decentralized execution (parameter sharing).
    - simulator: Simulator instance whose agents all share one policy.
    - update_fn: callable(buffer, returns) -> None that updates policy params.
                 Implement this with your DL framework (e.g., PPO/REINFORCE).
    """
    stats = []
    for ep in range(episodes):
        buf, returns = collect_episode(simulator, gamma=gamma)
        update_fn(buf, returns)
        ep_reward = sum(r for seq in returns.values() for r in seq) / max(len(returns), 1)
        stats.append({"episode": ep, "mean_return": ep_reward})
        buf.clear()
    return stats
