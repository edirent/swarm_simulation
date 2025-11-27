"""
Behavior cloning training script (Expert -> TinyMLP) with parameter sharing.
- Expert: heuristic Boids + target attraction + obstacle repulsion.
- Student: TinyMLP (PyTorch) driven by NNPolicy observation encoder.
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
from swarm.comms.network import BroadcastNetwork
from swarm.policies.rules_boirds import BoidsPolicy
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
        Target(center=t["center"], radius=t.get("radius", 0.3))
        for t in env_cfg.get("targets", [])
    ]
    return SwarmEnv(bounds=env_cfg["bounds"], obstacles=obstacles, targets=targets)


def build_agents(cfg, policy):
    agents = []
    bounds = cfg["env"]["bounds"]
    N = cfg["agents"]["count"]
    for i in range(N):
        st = AgentState(
            id=i,
            pos=np.random.uniform(bounds[0], bounds[1], size=2),
            vel=np.zeros(2),
            battery=1.0,
            role=0,
            task_id=None,
        )
        agents.append(Agent(st, policy))
    return agents


def expert_action(self_state, neighbor_msgs, env, boids_policy: BoidsPolicy, target_gain=1.0, obs_gain=1.5):
    # base boids
    base = boids_policy.act((self_state, neighbor_msgs))
    # target attraction
    target_term = np.zeros_like(self_state.vel)
    active = [t for t in env.targets if t.active]
    if active:
        ds = [np.linalg.norm(t.center - self_state.pos) for t in active]
        j = int(np.argmin(ds))
        vec = active[j].center - self_state.pos
        n = np.linalg.norm(vec) + 1e-6
        target_term = vec / n
    # obstacle repulsion
    obs_term = np.zeros_like(self_state.vel)
    for obs in env.obstacles:
        vec = self_state.pos - obs.center
        dist = np.linalg.norm(vec)
        if dist < obs.radius + 1.5:
            obs_term += vec / max(dist, 1e-6)
    return base + target_gain * target_term + obs_gain * obs_term


def collect_dataset(cfg, episodes=5, max_steps=500):
    device = get_device()
    boids = BoidsPolicy()
    student_obs_encoder = NNPolicy(
        model=TinyMLP(input_dim=NNPolicy.obs_dim(dim=2, k_max=cfg["policy"].get("k_max", 5))),
        k_max=cfg["policy"].get("k_max", 5),
        dim=2,
        device=device,
    )
    obs_buf: List[np.ndarray] = []
    act_buf: List[np.ndarray] = []
    dt = cfg["dt"]
    net = BroadcastNetwork(
        loss_prob=cfg["network"].get("loss_prob", 0.0),
        max_range=cfg["network"].get("max_range", None),
    )

    for _ in range(episodes):
        env = build_env(cfg)
        agents = build_agents(cfg, policy=None)  # policy not used; manual step below
        for step in range(max_steps):
            swarm_state = {i: a.state for i, a in enumerate(agents)}
            outgoing = [(i, a.to_message()) for i, a in enumerate(agents)]
            inbox = net.deliver(outgoing, type("SS", (), {"agents": swarm_state}))

            for i, agent in enumerate(agents):
                msgs = inbox[i]
                obs = student_obs_encoder.build_observation(agent.state, msgs, env.targets)
                act = expert_action(agent.state, msgs, env, boids)
                obs_buf.append(obs)
                act_buf.append(act.astype(np.float32))
                agent._apply_action(act, dt)

            # build SwarmState-like for constraints and collisions
            swarm_state_obj = type("SS", (), {"agents": swarm_state, "t": step * dt})
            env.enforce_constraints(swarm_state_obj)
            collisions, hits = env.check_collisions(swarm_state_obj)
            if env.all_targets_done():
                break
    X = np.stack(obs_buf).astype(np.float32)
    y = np.stack(act_buf).astype(np.float32)
    return X, y


def train_bc(cfg, X, y, hidden=32, lr=1e-3, epochs=10, batch_size=256, ckpt_path="checkpoints/tinymlp_bc.pt"):
    device = get_device()
    input_dim = X.shape[1]
    model = TinyMLP(input_dim=input_dim, hidden_dim=hidden, output_dim=y.shape[1]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    X_tensor = torch.from_numpy(X).to(device)
    y_tensor = torch.from_numpy(y).to(device)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            total += loss.item() * xb.size(0)
        avg = total / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - loss {avg:.6f}")

    ckpt = pathlib.Path(ckpt_path)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    print(f"Saved checkpoint to {ckpt}")
    return ckpt


def main():
    parser = argparse.ArgumentParser(description="Behavior cloning trainer.")
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("configs/base.yaml"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ckpt", type=pathlib.Path, default=pathlib.Path("checkpoints/tinymlp_bc.pt"))
    args = parser.parse_args()

    cfg = deep_update(
        {
            "dt": 0.1,
            "env": {"bounds": [-10, 10, -10, 10], "obstacles": [], "targets": []},
            "network": {"loss_prob": 0.0, "max_range": None},
            "policy": {"k_max": 5},
            "agents": {"count": 20},
        },
        load_config(args.config),
    )

    X, y = collect_dataset(cfg, episodes=args.episodes, max_steps=args.max_steps)
    train_bc(cfg, X, y, hidden=args.hidden, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, ckpt_path=args.ckpt)


if __name__ == "__main__":
    main()
