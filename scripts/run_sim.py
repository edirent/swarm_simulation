import argparse
import pathlib
import sys

import numpy as np
import yaml
import torch

# Ensure repository root is on PYTHONPATH when running without installation.
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
from swarm.viz.render_2d import SwarmRenderer2D
from swarm.viz.logger import SwarmLogger


DEFAULT_CONFIG = {
    "dt": 0.1,
    "steps": 5000,
    "render_every": 2,
    "env": {
        "bounds": [-20, 20, -20, 20],
        "sense_radius": 6.0,
        "enemies": [],
        "obstacles": [
            {"center": [-12, -5], "radius": 2.0},
            {"center": [-8, 8], "radius": 1.5},
            {"center": [-2, -10], "radius": 1.8},
            {"center": [3, 5], "radius": 2.0},
            {"center": [10, -6], "radius": 1.7},
            {"center": [12, 10], "radius": 2.2},
            {"center": [0, 12], "radius": 1.3},
        ],
        "targets": [
            {"center": [-15, 15], "radius": 0.5},
            {"center": [-10, 0], "radius": 0.5},
            {"center": [-2, 15], "radius": 0.5},
            {"center": [8, -12], "radius": 0.5},
            {"center": [15, 10], "radius": 0.5},
            {"center": [0, -15], "radius": 0.5},
        ],
    },
    "network": {"loss_prob": 0.1, "max_range": 5.0},
    "policy": {
        "type": "boids",  # boids | tinyml | nn
        "w_sep": 1.0,
        "w_align": 0.5,
        "w_coh": 0.5,
        "max_speed": 1.0,
        "hidden": 16,
        "k_max": 5,
        "checkpoint": None,
    },
    "agents": {"count": 30},
    "reward": {"hit": 10.0, "crash": -5.0, "step": -0.01, "approach": 0.1, "boundary": -10.0},
}


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
        return DEFAULT_CONFIG
    cfg = yaml.safe_load(path.read_text())
    if cfg is None:
        return DEFAULT_CONFIG
    if "inherits" in cfg:
        base_path = path.parent / cfg["inherits"]
        base_cfg = load_config(base_path)
        cfg = {k: v for k, v in cfg.items() if k != "inherits"}
        return deep_update(base_cfg, cfg)
    return deep_update(DEFAULT_CONFIG, cfg)


def make_policy(policy_cfg):
    if policy_cfg.get("type", "boids") == "boids":
        return BoidsPolicy(
            w_sep=policy_cfg.get("w_sep", 1.0),
            w_align=policy_cfg.get("w_align", 0.5),
            w_coh=policy_cfg.get("w_coh", 0.5),
            max_speed=policy_cfg.get("max_speed", 1.0),
        )
    if policy_cfg.get("type") == "nn":
        k_max = policy_cfg.get("k_max", 5)
        hidden = policy_cfg.get("hidden", 32)
        input_dim = NNPolicy.obs_dim(dim=2, k_max=k_max)
        device = get_device()
        model = TinyMLP(input_dim=input_dim, hidden_dim=hidden, output_dim=2)
        ckpt = policy_cfg.get("checkpoint")
        if ckpt:
            sd = torch.load(ckpt, map_location=device)
            model.load_state_dict(sd)
        return NNPolicy(model, k_max=k_max, dim=2, device=device)

    # tiny NumPy MLP fallback
    k_max = policy_cfg.get("k_max", 5)
    hidden = policy_cfg.get("hidden", 16)
    seed = policy_cfg.get("seed", 0)
    weights = TinyMLPPolicy.init_weights(dim=2, k_max=k_max, hidden=hidden, seed=seed)
    return TinyMLPPolicy(weights=weights, hidden=hidden, k_max=k_max, seed=seed, dim=2)


def build_agents(cfg, bounds):
    agents = []
    N = cfg["agents"]["count"]
    shared_policy = make_policy(cfg["policy"])
    setattr(shared_policy, "bounds", bounds)
    for i in range(N):
        st = AgentState(
            id=i,
            pos=np.random.uniform(bounds[0], bounds[1], size=2),
            vel=np.zeros(2),
            battery=1.0,
            role=0,
            task_id=None,
        )
        agents.append(Agent(st, shared_policy))
    return agents


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
    return SwarmEnv(
        bounds=env_cfg["bounds"],
        obstacles=obstacles,
        targets=targets,
        sense_radius=env_cfg.get("sense_radius", None),
        enemies=env_cfg.get("enemies", []),
    )


def main():
    parser = argparse.ArgumentParser(description="Run swarm simulation.")
    parser.add_argument("--config", type=pathlib.Path, help="Path to YAML config.")
    parser.add_argument("--no-render", action="store_true", help="Disable live rendering (headless).")
    parser.add_argument("--log", type=pathlib.Path, help="Optional path to write JSON log.")
    parser.add_argument("--steps", type=int, help="Override total simulation steps.")
    parser.add_argument("--dt", type=float, help="Override simulation timestep.")
    parser.add_argument("--render-every", type=int, dest="render_every", help="Render every N steps.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.dt is not None:
        cfg["dt"] = args.dt
    if args.render_every is not None:
        cfg["render_every"] = args.render_every
    bounds = cfg["env"]["bounds"]

    agents = build_agents(cfg, bounds)
    env = build_env(cfg)
    net = BroadcastNetwork(
        loss_prob=cfg["network"].get("loss_prob", 0.0),
        max_range=cfg["network"].get("max_range", None),
    )
    sim = Simulator(agents, env, net, dt=cfg["dt"], reward_cfg=cfg.get("reward"))
    renderer = None if args.no_render else SwarmRenderer2D(bounds=bounds, obstacles=env.obstacles, targets=env.targets)
    logger = SwarmLogger(args.log) if args.log else None

    for step in range(cfg["steps"]):
        state, rewards, collisions, done, hits = sim.step()
        if renderer and step % cfg["render_every"] == 0:
            renderer.render(state, enemies=list(sim.enemies.values()))
        if logger:
            logger.log_state(state)
        if done:
            print(f"All targets collected at step {step}")
            break

    if logger:
        logger.flush()


if __name__ == "__main__":
    main()
