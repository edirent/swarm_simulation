"""
Alternating self-play (freeze one side, train the other).
Generation loop:
  - Gen 0: frozen red (hand-crafted) vs trainable blue (NN)
  - Gen 1: freeze blue snapshot -> train red vs frozen blue
  - Gen 2+: continue alternating (blue vs frozen red, etc.)
"""

import argparse
import copy
import pathlib
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swarm.core.state import AgentState
from swarm.core.agent import Agent
from swarm.core.env import SwarmEnv, Obstacle, Target
from swarm.core.simulator import Simulator
from swarm.comms.network import BroadcastNetwork
from swarm.policies.rules_boirds import BoidsPolicy
from swarm.policies.tinyml_mlp import TinyMLPPolicy
from swarm.policies.nn_mlp import TinyMLP, NNPolicy, get_device

TEAM_NAME_TO_ID = {"blue": 0, "red": 1}


def deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: pathlib.Path | None) -> dict:
    if path is None:
        return {}
    cfg = yaml.safe_load(path.read_text()) or {}
    if "inherits" in cfg:
        base_path = path.parent / cfg["inherits"]
        base_cfg = load_config(base_path)
        cfg = {k: v for k, v in cfg.items() if k != "inherits"}
        return deep_update(base_cfg, cfg)
    return cfg


def build_env(cfg):
    env_cfg = cfg.get("env", {})
    obstacles = [
        Obstacle(center=o["center"], radius=o.get("radius", 1.0))
        for o in env_cfg.get("obstacles", [])
    ]
    targets = [
        Target(center=t["center"], radius=t.get("radius", 0.5))
        for t in env_cfg.get("targets", [])
    ]
    return SwarmEnv(
        bounds=env_cfg["bounds"],
        obstacles=obstacles,
        targets=targets,
        sense_radius=env_cfg.get("sense_radius", None),
        enemies=env_cfg.get("enemies", []),
        resource_cfg=cfg.get("resource", {}),
    )


def make_policy(policy_cfg: dict, bounds, device):
    ptype = policy_cfg.get("type", "boids")
    if ptype == "boids":
        policy = BoidsPolicy(
            w_sep=policy_cfg.get("w_sep", 1.0),
            w_align=policy_cfg.get("w_align", 0.5),
            w_coh=policy_cfg.get("w_coh", 0.5),
            w_target=policy_cfg.get("w_target", 0.3),
            w_resource=policy_cfg.get("w_resource", 0.8),
            max_speed=policy_cfg.get("max_speed", 1.0),
        )
        setattr(policy, "bounds", bounds)
        return policy
    if ptype == "nn":
        k_max = policy_cfg.get("k_max", 5)
        hidden = policy_cfg.get("hidden", 64)
        input_dim = NNPolicy.obs_dim(dim=2, k_max=k_max)
        model = TinyMLP(input_dim=input_dim, hidden_dim=hidden, output_dim=2)
        ckpt = policy_cfg.get("checkpoint")
        if ckpt:
            sd = torch.load(ckpt, map_location=device)
            model.load_state_dict(sd)
        policy = NNPolicy(model, k_max=k_max, dim=2, device=device, stochastic=True, noise_std=policy_cfg.get("noise_std", 0.2))
        setattr(policy, "bounds", bounds)
        return policy
    # lightweight NumPy fallback
    k_max = policy_cfg.get("k_max", 5)
    hidden = policy_cfg.get("hidden", 16)
    seed = policy_cfg.get("seed", 0)
    weights = TinyMLPPolicy.init_weights(dim=2, k_max=k_max, hidden=hidden, seed=seed)
    policy = TinyMLPPolicy(weights=weights, hidden=hidden, k_max=k_max, seed=seed, dim=2)
    setattr(policy, "bounds", bounds)
    return policy


def clone_policy(policy):
    return copy.deepcopy(policy)


def build_team_id_map(teams_cfg: dict) -> Dict[str, int]:
    name_to_id = dict(TEAM_NAME_TO_ID)
    for name in teams_cfg.keys():
        if name not in name_to_id:
            name_to_id[name] = len(name_to_id)
    return name_to_id


def build_agents_with_policies(cfg, bounds, policy_map: Dict[str, object], name_to_id: Dict[str, int]):
    agents = []
    next_id = 0
    teams_cfg = cfg.get("teams", {})
    for team_name, tcfg in teams_cfg.items():
        policy = policy_map[team_name]
        team_id = name_to_id[team_name]
        count = tcfg.get("count", 0)
        for _ in range(count):
            st = AgentState(
                id=next_id,
                pos=np.random.uniform(bounds[0], bounds[1], size=2),
                vel=np.zeros(2),
                battery=1.0,
                role=0,
                task_id=None,
                team=team_id,
            )
            agents.append(Agent(st, policy))
            next_id += 1
    return agents


def rollout_episode(cfg, policy_map: Dict[str, object], train_team: str, name_to_id: Dict[str, int], max_steps: int, gamma: float, noise_std: float, device) -> Tuple[np.ndarray | None, torch.Tensor | None, torch.Tensor | None, Dict[int, int]]:
    bounds = cfg["env"]["bounds"]
    env = build_env(cfg)
    agents = build_agents_with_policies(cfg, bounds, policy_map, name_to_id)
    net = BroadcastNetwork(
        loss_prob=cfg["network"].get("loss_prob", 0.0),
        max_range=cfg["network"].get("max_range", None),
    )
    sim = Simulator(agents, env, net, dt=cfg["dt"], reward_cfg=cfg.get("reward"))
    train_team_id = name_to_id[train_team]
    train_policy = policy_map[train_team]
    if isinstance(train_policy, NNPolicy):
        train_policy.stochastic = True
        train_policy.noise_std = noise_std
    obs_buf, logprob_buf, rew_buf = [], [], []

    for _ in range(max_steps):
        _, rewards, _, done, _, logs = sim.step(return_logs=True)
        for aid, log in logs.items():
            if not isinstance(log, dict) or "obs" not in log:
                continue
            if aid not in sim.agents:
                continue
            if sim.agents[aid].state.team != train_team_id:
                continue
            if not hasattr(train_policy, "log_prob"):
                continue
            logprob_buf.append(train_policy.log_prob(log["obs"], log["action"]))
            obs_buf.append(log["obs"].astype(np.float32))
            rew_buf.append(rewards.get(aid, 0.0))
        if done:
            break

    if not logprob_buf:
        return None, None, None, sim.team_scores

    returns = []
    G = 0.0
    for r in reversed(rew_buf):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
    logprob_t = torch.stack(logprob_buf).to(device)
    obs_np = np.stack(obs_buf)
    return obs_np, logprob_t, returns_t, sim.team_scores


def train_team_policy(cfg, train_team: str, policy_map: Dict[str, object], name_to_id: Dict[str, int], episodes: int, max_steps: int, gamma: float, noise_std: float, lr: float, device):
    train_policy = policy_map[train_team]
    if not isinstance(train_policy, NNPolicy):
        print(f"[skip] Team {train_team} policy is not trainable (type={type(train_policy).__name__}).")
        return []
    optimizer = torch.optim.Adam(train_policy.model.parameters(), lr=lr)
    stats = []
    for ep in range(episodes):
        obs_np, logprob_t, returns_t, scores = rollout_episode(
            cfg, policy_map, train_team, name_to_id, max_steps=max_steps, gamma=gamma, noise_std=noise_std, device=device
        )
        if logprob_t is None:
            print(f"[warn] No samples collected for team {train_team} in episode {ep}.")
            continue
        adv = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        loss = -(logprob_t * adv.to(device)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stats.append({"episode": ep, "loss": float(loss.item()), "scores": scores})
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[train] Team {train_team} ep {ep+1}/{episodes}, loss={loss.item():.4f}, scores={scores}")
    return stats


def alternating_self_play(cfg, generations: int, episodes_per_gen: int, max_steps: int, gamma: float, lr: float, noise_std: float, outdir: pathlib.Path):
    device = get_device()
    teams_cfg = cfg.get("teams", {})
    if not teams_cfg:
        raise ValueError("Self-play requires cfg['teams'] with at least blue/red entries.")
    name_to_id = build_team_id_map(teams_cfg)
    bounds = cfg["env"]["bounds"]

    trainable = {name: make_policy(tcfg.get("policy", cfg.get("policy", {})), bounds, device) for name, tcfg in teams_cfg.items()}
    frozen_snapshots = {
        name: make_policy(tcfg.get("frozen_policy", tcfg.get("opponent_policy", tcfg.get("policy", {"type": "boids"}))), bounds, device)
        for name, tcfg in teams_cfg.items()
    }

    outdir.mkdir(parents=True, exist_ok=True)

    for gen in range(generations):
        train_team = "blue" if gen % 2 == 0 else "red"
        policy_map = dict(frozen_snapshots)
        policy_map[train_team] = trainable[train_team]
        print(f"\n=== Generation {gen}: train {train_team} vs frozen opponent ===")
        stats = train_team_policy(
            cfg,
            train_team=train_team,
            policy_map=policy_map,
            name_to_id=name_to_id,
            episodes=episodes_per_gen,
            max_steps=max_steps,
            gamma=gamma,
            noise_std=noise_std,
            lr=lr,
            device=device,
        )
        # snapshot for the next round
        frozen_snapshots[train_team] = clone_policy(trainable[train_team])
        if isinstance(trainable[train_team], NNPolicy):
            ckpt = outdir / f"{train_team}_gen{gen}.pt"
            torch.save(trainable[train_team].model.state_dict(), ckpt)
            print(f"[ckpt] saved {ckpt}")
    return trainable, frozen_snapshots


def main():
    parser = argparse.ArgumentParser(description="Alternating self-play trainer.")
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("configs/self_play.yaml"))
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("checkpoints/self_play"))
    args = parser.parse_args()

    base_cfg = {
        "dt": 0.1,
        "env": {"bounds": [-20, 20, -20, 20], "obstacles": [], "targets": [], "sense_radius": 6.0},
        "network": {"loss_prob": 0.0, "max_range": None},
        "policy": {"type": "nn", "k_max": 5, "hidden": 64},
        "teams": {
            "blue": {"count": 12, "policy": {"type": "nn", "k_max": 5, "hidden": 64}},
            "red": {
                "count": 12,
                "policy": {"type": "nn", "k_max": 5, "hidden": 64},
                "frozen_policy": {"type": "boids", "w_sep": 1.2, "w_resource": 1.2, "w_target": 0.2},
            },
        },
        "resource": {"enabled": True, "radius": 0.7, "margin": 1.0, "seed": 0},
        "reward": {"hit": 10.0, "crash": -5.0, "boundary": -10.0, "step": -0.01, "approach": 0.1, "resource": 8.0},
    }
    cfg = deep_update(base_cfg, load_config(args.config))
    alternating_self_play(
        cfg,
        generations=args.generations,
        episodes_per_gen=args.episodes,
        max_steps=args.max_steps,
        gamma=args.gamma,
        lr=args.lr,
        noise_std=args.noise_std,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
