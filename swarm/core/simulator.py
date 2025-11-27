import numpy as np

from .state import SwarmState, AgentState
from .agent import Agent
from .enemy import Enemy
from ..comms.network import BroadcastNetwork


class Simulator:
    def __init__(self, agents: list[Agent], env, network: BroadcastNetwork, dt=0.1, reward_cfg=None):
        self.agents = {a.state.id: a for a in agents}
        self.enemies: dict[int, Enemy] = self._init_enemies(env.enemy_spawns if hasattr(env, "enemy_spawns") else [], env.bounds)
        self.env = env
        self.network = network
        self.dt = dt
        self.t = 0.0
        self.reward_cfg = reward_cfg or {
            "hit": 10.0,
            "crash": -5.0,
            "step": -0.01,
            "approach": 0.1,
            "boundary": -2.0,
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
            if self.env.sense_radius is not None:
                msgs = [m for m in msgs if np.linalg.norm(m.pos - agent.state.pos) <= self.env.sense_radius]
            visible_targets = self.env.visible_targets(agent.state)
            obs, action = agent.step(msgs, visible_targets, self.dt)
            step_logs[i] = {"obs": obs, "action": action}

        swarm_state = SwarmState(
            agents={i: self._clone_agent_state(a.state) for i, a in self.agents.items()},
            t=self.t + self.dt,
        )
        boundary_hits = self.env.enforce_constraints(swarm_state)

        enemy_contacts = self._step_enemies_and_contacts(swarm_state)
        collisions, hits = self.env.check_collisions(swarm_state)
        rewards = self.compute_rewards(prev_state, swarm_state, collisions, hits, boundary_hits, enemy_contacts)

        for i, st in swarm_state.agents.items():
            self.agents[i].state = st
        self._prune_dead_agents(enemy_contacts)

        self.t += self.dt
        done = self.env.all_targets_done()
        if return_logs:
            for aid, hit_boundary in boundary_hits.items():
                step_logs[aid]["boundary_hit"] = hit_boundary
            step_logs["enemy_contacts"] = enemy_contacts
            return swarm_state, rewards, collisions, done, hits, step_logs
        return swarm_state, rewards, collisions, done, hits

    def compute_rewards(self, prev_state, new_state, collisions, hits, boundary_hits=None, enemy_contacts=None):
        rewards = {i: 0.0 for i in new_state.agents.keys()}
        R_HIT = self.reward_cfg.get("hit", 10.0)
        R_CRASH = self.reward_cfg.get("crash", -5.0)
        R_STEP = self.reward_cfg.get("step", -0.01)
        R_APPROACH = self.reward_cfg.get("approach", 0.1)
        R_BOUNDARY = self.reward_cfg.get("boundary", -2.0)
        enemy_contacts = enemy_contacts or []
        boundary_hits = boundary_hits or {i: False for i in new_state.agents.keys()}

        for i in rewards.keys():
            rewards[i] += R_STEP

        for i, hit_list in hits.items():
            if len(hit_list) > 0:
                rewards[i] += R_HIT * len(hit_list)

        for i, crashed in collisions.items():
            if crashed:
                rewards[i] += R_CRASH

        for aid, _ in enemy_contacts:
            if aid in rewards:
                rewards[aid] += R_CRASH + R_HIT

        for i, hit_boundary in boundary_hits.items():
            if hit_boundary:
                rewards[i] += R_BOUNDARY

        for i, st in new_state.agents.items():
            if not self.env.targets:
                continue
            d_min_before = self._min_target_dist(prev_state.agents[i].pos)
            d_min_after = self._min_target_dist(st.pos)
            delta = d_min_before - d_min_after
            if delta > 0:
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

    def _init_enemies(self, enemy_cfg_list, bounds):
        enemies = {}
        for idx, cfg in enumerate(enemy_cfg_list):
            e = Enemy(
                id=-(idx + 1),  # negative IDs to distinguish
                pos=cfg.get("pos", [0, 0]),
                speed=cfg.get("speed", 1.2),
                sense=cfg.get("sense", 8.0),
                radius=cfg.get("radius", 0.6),
            )
            e._clamp(bounds)
            enemies[e.state.id] = e
        return enemies

    def _step_enemies_and_contacts(self, swarm_state):
        contacts = []
        alive_agents = [a for a in swarm_state.agents.values() if a.alive]
        for enemy in self.enemies.values():
            if not enemy.state.alive:
                continue
            enemy.step(alive_agents, self.dt, self.env.bounds)
            for agent in alive_agents:
                d = np.linalg.norm(agent.pos - enemy.state.pos)
                if d <= enemy.radius:
                    contacts.append((agent.id, enemy.state.id))
                    agent.alive = False
                    enemy.state.alive = False
        return contacts

    def _prune_dead_agents(self, enemy_contacts):
        dead_ids = {aid for aid, _ in enemy_contacts}
        for aid in list(dead_ids):
            self.agents.pop(aid, None)
