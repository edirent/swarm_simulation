from .state import SwarmState
from .agent import Agent
from ..comms.network import BroadcastNetwork


class Simulator:
    def __init__(self, agents: list[Agent], env, network: BroadcastNetwork, dt=0.1):
        self.agents = {a.state.id: a for a in agents}
        self.env = env
        self.network = network
        self.dt = dt
        self.t = 0.0

    def step(self):
        swarm_state = SwarmState(
            agents={i: a.state for i, a in self.agents.items()},
            t=self.t
        )

        # 1) each agent forms its broadcast msg
        outgoing = [(i, a.to_message()) for i, a in self.agents.items()]

        # 2) network delivers messages (broadcast model)
        inbox = self.network.deliver(outgoing, swarm_state)

        # 3) each agent updates based on its neighbor msgs
        for i, agent in self.agents.items():
            msgs = inbox[i]
            agent.step(msgs, self.dt)

        # 4) enforce environment constraints
        swarm_state = SwarmState(
            agents={i: a.state for i, a in self.agents.items()},
            t=self.t + self.dt
        )
        self.env.enforce_constraints(swarm_state)

        # sync back
        for i, st in swarm_state.agents.items():
            self.agents[i].state = st

        self.t += self.dt
        return swarm_state
