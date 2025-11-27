from .state import AgentState
from ..policies.base import Policy
from ..comms.messages import SwarmMessage


class Agent:
    def __init__(self, state: AgentState, policy: Policy):
        self.state = state
        self.policy = policy

    def step(self, neighbor_msgs: list[SwarmMessage], dt: float, targets=None):
        """
        Single-threaded, no locks. All perception + decision here.
        """
        obs = self.policy.build_observation(self.state, neighbor_msgs, targets=targets)
        action = self.policy.act(obs)  # typically target velocity / accel
        self._apply_action(action, dt)
        return obs, action

    def _apply_action(self, action, dt: float):
        # e.g., action = desired velocity delta
        self.state.vel += action
        self.state.pos += self.state.vel * dt

    def to_message(self) -> SwarmMessage:
        # convert current state to broadcast message
        from ..comms.messages import SwarmMessage
        return SwarmMessage.from_state(self.state)
