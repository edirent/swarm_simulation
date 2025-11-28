import json
from pathlib import Path
from ..core.state import SwarmState


class SwarmLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = []

    def log_state(self, state: SwarmState, scores=None, resource=None):
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
                    "team": st.team,
                }
                for aid, st in state.agents.items()
            },
        }
        if scores is not None:
            snapshot["scores"] = scores
        if resource is not None:
            snapshot["resource"] = {"center": resource.center.tolist(), "radius": resource.radius}
        self.records.append(snapshot)

    def flush(self):
        with self.path.open("w") as f:
            json.dump(self.records, f, indent=2)
