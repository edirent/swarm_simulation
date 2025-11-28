"""
On-policy REINFORCE training with shared TinyMLP policy (Gaussian exploration).
Defaults: 500 episodes, macOS/MPS friendly.
"""

import argparse
import pathlib
import sys
from typing import List

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
from swarm.policies.nn_mlp import TinyMLP, NNPolicy, get_device


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


def build_agents(cfg, policy):
    agents = []
    bounds = cfg["env"]["bounds"]
    setattr(policy, "bounds", bounds)
    N = cfg["agents"]["count"]
    for i in range(N):
        st = AgentState(
            id=i,
            pos=np.random.uniform(bounds[0], bounds[1], size=2),
            vel=np.zeros(2),
            battery=1.0,
            role=0,
            task_id=None,
            team=0,
        )
        agents.append(Agent(st, policy))
    return agents


def rollout_episode(cfg, policy, max_steps, gamma, noise_std=0.2):
    # reset env / agents each episode
    env = build_env(cfg)
    policy.noise_std = noise_std
    policy.stochastic = True
    agents = build_agents(cfg, policy)
    net = BroadcastNetwork(
        loss_prob=cfg["network"].get("loss_prob", 0.0),
        max_range=cfg["network"].get("max_range", None),
    )
    sim = Simulator(agents, env, net, dt=cfg["dt"], reward_cfg=cfg.get("reward"))

    obs_buf: List[np.ndarray] = []
    act_buf: List[np.ndarray] = []
    logprob_buf: List[torch.Tensor] = []
    rew_buf: List[float] = []

    for _ in range(max_steps):
        state, rewards, collisions, done, hits, logs = sim.step(return_logs=True)
        for aid, log in logs.items():
            if not isinstance(log, dict) or "obs" not in log:
                continue
            obs = log["obs"]
            act = log["action"]
            obs_buf.append(obs.astype(np.float32))
            act_buf.append(act.astype(np.float32))
            logprob_buf.append(policy.log_prob(obs, act))
            rew_buf.append(rewards.get(aid, 0.0))
        if done:
            break

    # compute returns (flattened over agents and time)
    returns = []
    G = 0.0
    for r in reversed(rew_buf):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    returns_t = torch.tensor(returns, dtype=torch.float32, device=policy.device)
    logprob_t = torch.stack(logprob_buf).to(policy.device)
    obs_np = np.stack(obs_buf)
    act_np = np.stack(act_buf)
    return obs_np, act_np, logprob_t, returns_t


def reinforce_train(cfg, episodes=500, max_steps=400, gamma=0.99, lr=1e-3, hidden=32, noise_std=0.2, ckpt="checkpoints/rl_policy.pt"):
    device = get_device()
    input_dim = NNPolicy.obs_dim(dim=2, k_max=cfg["policy"].get("k_max", 5))
    model = TinyMLP(input_dim=input_dim, hidden_dim=hidden, output_dim=2).to(device)
    policy = NNPolicy(model, k_max=cfg["policy"].get("k_max", 5), dim=2, device=device, stochastic=True, noise_std=noise_std)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(episodes):
        obs_np, act_np, logprob_t, returns_t = rollout_episode(cfg, policy, max_steps=max_steps, gamma=gamma, noise_std=noise_std)
        # normalize returns for stability
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        loss = -(logprob_t * returns_t.to(device)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[EP {ep+1}/{episodes}] loss={loss.item():.4f}, steps={len(obs_np)}")

    ckpt_path = pathlib.Path(ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved RL policy to {ckpt_path}")
    return ckpt_path


def main():
    parser = argparse.ArgumentParser(description="REINFORCE trainer (shared policy).")
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("configs/base.yaml"))
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--ckpt", type=pathlib.Path, default=pathlib.Path("checkpoints/rl_policy.pt"))
    args = parser.parse_args()

    base_cfg = {
        "dt": 0.1,
        "env": {"bounds": [-20, 20, -20, 20], "obstacles": [], "targets": [], "sense_radius": 6.0},
        "network": {"loss_prob": 0.0, "max_range": None},
        "policy": {"k_max": 5},
        "agents": {"count": 30},
        "reward": {"hit": 10.0, "crash": -5.0, "boundary": -10.0, "step": -0.01, "approach": 0.1},
    }
    cfg = deep_update(base_cfg, load_config(args.config))
    reinforce_train(
        cfg,
        episodes=args.episodes,
        max_steps=args.max_steps,
        gamma=args.gamma,
        lr=args.lr,
        hidden=args.hidden,
        noise_std=args.noise_std,
        ckpt=args.ckpt,
    )


if __name__ == "__main__":
    main()
