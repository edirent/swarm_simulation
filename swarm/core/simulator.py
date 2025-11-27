from .state import SwarmState, AgentState
from .agent import Agent
from ..comms.network import BroadcastNetwork


class Simulator:
    def __init__(self, agents: list[Agent], env, network: BroadcastNetwork, dt=0.1, reward_cfg=None):
        self.agents = {a.state.id: a for a in agents}
        self.env = env
        self.network = network
        self.dt = dt
        self.t = 0.0
        self.reward_cfg = reward_cfg or {
            "hit": 10.0,
            "crash": -5.0,
            "step": -0.01,
            "approach": 0.1,
        }

    def step(self, return_logs: bool = False):
        prev_state = SwarmState(
            agents={i: self._clone_agent_state(a.state) for i, a in self.agents.items()},
            t=self.t,
        )

        outgoing = [(i, a.to_message()) for i, a in self.agents.items()]
        inbox = self.network.deliver(outgoing, prev_state)

        step_logs = {}
        for i, agent in self.agents.items():
            msgs = inbox[i]
            obs, action = agent.step(msgs, self.dt, targets=self.env.targets)
            step_logs[i] = {"obs": obs, "action": action}

        swarm_state = SwarmState(
            agents={i: self._clone_agent_state(a.state) for i, a in self.agents.items()},
            t=self.t + self.dt,
        )
        self.env.enforce_constraints(swarm_state)

        collisions, hits = self.env.check_collisions(swarm_state)
        rewards = self.compute_rewards(prev_state, swarm_state, collisions, hits)

        for i, st in swarm_state.agents.items():
            self.agents[i].state = st

        self.t += self.dt
        done = self.env.all_targets_done()
        if return_logs:
            return swarm_state, rewards, collisions, done, hits, step_logs
        return swarm_state, rewards, collisions, done, hits

    def compute_rewards(self, prev_state, new_state, collisions, hits):
        rewards = {i: 0.0 for i in new_state.agents.keys()}
        R_HIT = self.reward_cfg.get("hit", 10.0)
        R_CRASH = self.reward_cfg.get("crash", -5.0)
        R_STEP = self.reward_cfg.get("step", -0.01)
        R_APPROACH = self.reward_cfg.get("approach", 0.1)

        for i in rewards.keys():
            rewards[i] += R_STEP

        for i, hit_list in hits.items():
            if len(hit_list) > 0:
                rewards[i] += R_HIT * len(hit_list)

        for i, crashed in collisions.items():
            if crashed:
                rewards[i] += R_CRASH

        for i, st in new_state.agents.items():
            if not self.env.targets:
                continue
            d_min_before = self._min_target_dist(prev_state.agents[i].pos)
            d_min_after = self._min_target_dist(st.pos)
            delta = d_min_before - d_min_after
            rewards[i] += R_APPROACH * delta

        return rewards

    def _min_target_dist(self, pos):
        ds = []
        for tgt in self.env.targets:
            if not tgt.active:
                continue
            ds.append(((pos - tgt.center) ** 2).sum() ** 0.5)
        return min(ds) if ds else 0.0

    def _clone_agent_state(self, st: AgentState) -> AgentState:
        return AgentState(
            id=st.id,
            pos=st.pos.copy(),
            vel=st.vel.copy(),
            battery=st.battery,
            role=st.role,
            task_id=st.task_id,
            alive=st.alive,
        )
