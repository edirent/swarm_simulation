import argparse
import pathlib
import sys

import numpy as np
import yaml

# Ensure repository root is on PYTHONPATH when running without installation.
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swarm.core.state import AgentState
from swarm.core.agent import Agent
from swarm.core.env import SwarmEnv
from swarm.core.simulator import Simulator
from swarm.comms.network import BroadcastNetwork
from swarm.policies.rules_boirds import BoidsPolicy
from swarm.policies.tinyml_mlp import TinyMLPPolicy
from swarm.viz.render_2d import SwarmRenderer2D
from swarm.viz.logger import SwarmLogger


DEFAULT_CONFIG = {
    "dt": 0.1,
    "steps": 1000,
    "render_every": 2,
    "env": {"bounds": [-10, 10, -10, 10]},
    "network": {"loss_prob": 0.1, "max_range": 5.0},
    "policy": {"type": "boids", "w_sep": 1.0, "w_align": 0.5, "w_coh": 0.5, "max_speed": 1.0},
    "agents": {"count": 20},
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
    # TinyML default: single hidden layer with random weights sized for 2D + k_max=5
    input_dim = 26  # see TinyMLPPolicy.build_observation for 2D positions
    hidden = policy_cfg.get("hidden", 16)
    rng = np.random.default_rng(seed=policy_cfg.get("seed", 0))
    weights = {
        "W1": rng.standard_normal((hidden, input_dim)) * 0.1,
        "b1": np.zeros(hidden),
        "W2": rng.standard_normal((2, hidden)) * 0.1,
        "b2": np.zeros(2),
    }
    return TinyMLPPolicy(weights)


def build_agents(cfg, bounds):
    agents = []
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
        policy = make_policy(cfg["policy"])
        agents.append(Agent(st, policy))
    return agents


def main():
    parser = argparse.ArgumentParser(description="Run swarm simulation.")
    parser.add_argument("--config", type=pathlib.Path, help="Path to YAML config.")
    parser.add_argument("--no-render", action="store_true", help="Disable live rendering (headless).")
    parser.add_argument("--log", type=pathlib.Path, help="Optional path to write JSON log.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bounds = cfg["env"]["bounds"]

    agents = build_agents(cfg, bounds)
    env = SwarmEnv(bounds=bounds)
    net = BroadcastNetwork(
        loss_prob=cfg["network"].get("loss_prob", 0.0),
        max_range=cfg["network"].get("max_range", None),
    )
    sim = Simulator(agents, env, net, dt=cfg["dt"])
    renderer = None if args.no_render else SwarmRenderer2D(bounds=bounds)
    logger = SwarmLogger(args.log) if args.log else None

    for step in range(cfg["steps"]):
        state = sim.step()
        if renderer and step % cfg["render_every"] == 0:
            renderer.render(state)
        if logger:
            logger.log_state(state)

    if logger:
        logger.flush()


if __name__ == "__main__":
    main()
