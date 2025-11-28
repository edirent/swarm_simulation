from dataclasses import dataclass
import numpy as np
from ..core.state import AgentState


@dataclass
class SwarmMessage:
    sender_id: int
    pos: np.ndarray
    vel: np.ndarray
    battery: float
    role: int
    task_id: int | None
    t: float
    team: int

    @classmethod
    def from_state(cls, state: AgentState):
        return cls(
            sender_id=state.id,
            pos=state.pos.copy(),
            vel=state.vel.copy(),
            battery=state.battery,
            role=state.role,
            task_id=state.task_id,
            t=0.0,  # overwritten by simulator time if needed
            team=state.team,
        )
