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
Boids-style swarm在有障碍（圆形树木）和目标的地图上飞行，命中目标得分、撞树扣分。加上/去掉 `--no-render` 控制是否可视化。

### CLI options
- `--config configs/base.yaml` to load a YAML config (supports `inherits: base.yaml` for overrides).
- `--no-render` to run headless.
- `--log logs/sim.json` to dump states for later plotting.
- `--steps`, `--dt`, `--render-every` 覆盖配置文件里的步数、时间步长、渲染频率。
- `policy.type` 可选 `boids` / `tinyml` / `nn`（PyTorch TinyMLP，可用 `policy.checkpoint` 加载权重）。

## Notes
- `Simulator` is the deterministic core. Each agent owns its state + policy; no shared memory.
- `BroadcastNetwork` encapsulates loss/range; swap for real UDP/mcast later without touching agent logic.
- `Policy` interface stays stable across hand-crafted rules, tiny MLP, or future MARL；tiny MLP 观测里包含邻居信息 + 最近目标相对位置。
- 环境支持圆形障碍/目标、边界约束、命中判定、基础奖励（命中奖励、撞击惩罚、靠近奖励、step 惩罚）。
- `swarm/training/rl_train.py` 给出 CTDE（集中训练、分布执行）的最小训练 loop 骨架，便于接入你自己的优化器/深度学习库。
- `scripts/train_bc.py` 给了行为克隆示例：用启发式 Boids+目标/避障 采样专家动作，训练 PyTorch TinyMLP，并保存权重到 `checkpoints/tinymlp_bc.pt`，可在 `run_sim.py` 中通过 `policy.type=nn` + `policy.checkpoint` 加载。

## Mac Studio（MPS）快速指南
1) 确认原生 arm64 终端：`python -c "import platform; print(platform.machine())"`（应输出 arm64）。
2) 安装 PyTorch（包含 MPS）：`pip install "torch>=2.3" --index-url https://download.pytorch.org/whl/cpu`
3) 设备选择：`swarm/policies/nn_mlp.py` 使用 `get_device()` 自动选 `mps` 优先，其次 `cpu`。

### 三步打通“仿真-训练-部署”
- Step 1：`policy.type=nn`，无 checkpoint 时用随机 TinyMLP + NNPolicy 跑 `scripts/run_sim.py`，验证 PyTorch+MPS+仿真链路。
- Step 2：运行 `python scripts/train_bc.py --config configs/base.yaml --episodes 5 --epochs 10` 用专家 (Boids+目标吸引+避障) 行为克隆，生成 `checkpoints/tinymlp_bc.pt`。
- Step 3：在 `configs/base.yaml`（或 CLI 覆盖）设置 `policy.type: nn`、`policy.checkpoint: checkpoints/tinymlp_bc.pt`，再跑 `scripts/run_sim.py`，观察不依赖专家的 TinyMLP 行为。
