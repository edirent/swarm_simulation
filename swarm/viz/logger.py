import json
from pathlib import Path
from ..core.state import SwarmState


class SwarmLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = []

    def log_state(self, state: SwarmState):
        snapshot = {
            "t": state.t,
            "agents": {
                aid: {
                    "pos": st.pos.tolist(),
                    "vel": st.vel.tolist(),
                    "battery": st.battery,
                    "role": st.role,
                    "task_id": st.task_id,
                    "alive": st.alive,
                }
                for aid, st in state.agents.items()
            },
        }
        self.records.append(snapshot)

    def flush(self):
        with self.path.open("w") as f:
            json.dump(self.records, f, indent=2)
