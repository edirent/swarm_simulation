from dataclasses import dataclass
import numpy as np


@dataclass
class AgentState:
    id: int
    pos: np.ndarray      # shape (2,) or (3,)
    vel: np.ndarray      # same dim as pos
    battery: float       # 0..1
    role: int            # 0=normal, 1=leader, etc.
    task_id: int | None
    alive: bool = True


@dataclass
class SwarmState:
    agents: dict[int, AgentState]
    t: float
