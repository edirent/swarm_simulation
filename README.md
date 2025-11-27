# Swarm TinyML Simulation

Minimal swarm simulation playground with pluggable policies (rules or tiny ML), broadcast-style comms, and simple visualization.

## Layout
```
swarm-tinyml-sim/
├── configs/                 # YAML configs for quick sweeps
├── swarm/                   # core library
│   ├── core/                # Agent/env/simulator state and loop
│   ├── comms/               # Broadcast network + message schema
│   ├── policies/            # Boids rules + tiny MLP policy
│   ├── tasks/               # Coverage / target tracking stubs
│   ├── training/            # Dataset + RL placeholder
│   └── viz/                 # Matplotlib renderer + logger
└── scripts/                 # Entrypoints (run_sim.py, sweeps, plotting)
```

## Quickstart
```
pip install -r requirements.txt
python scripts/run_sim.py --no-render
```
Renders a simple Boids-style swarm inside bounds (add/remove `--no-render` for live matplotlib). Edit `configs/*.yaml` or swap policies/tasks as you extend.

### CLI options
- `--config configs/base.yaml` to load a YAML config (supports `inherits: base.yaml` for overrides).
- `--no-render` to run headless.
- `--log logs/sim.json` to dump states for later plotting.

## Notes
- `Simulator` is the deterministic core. Each agent owns its state + policy; no shared memory.
- `BroadcastNetwork` encapsulates loss/range; swap for real UDP/mcast later without touching agent logic.
- `Policy` interface stays stable across hand-crafted rules, tiny MLP, or future MARL.
