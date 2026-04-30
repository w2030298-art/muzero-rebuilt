# 开发计划

## 元信息

| 项 | 内容 |
|---|---|
| 项目名称 | MuZero Rebuilt |
| 项目方式 | 从零重构 |
| 目标硬件 | RTX 4060 Laptop，默认按单 GPU、8GB 显存级别约束 |
| Python 版本 | Python 3.11 |
| 深度学习框架 | PyTorch 2.x |
| 环境 API | Gymnasium |
| 配置系统 | Pydantic v2 + PyYAML |
| CLI | Typer |
| 包管理 | uv + `pyproject.toml` |
| Lint / Format | ruff |
| 类型检查 | pyright |
| 测试 | pytest |
| 日志 | Python logging + TensorBoard |
| 分布式 | 默认 Local Executor；Ray optional extra |
| 总模块数 | 18 |
| 预计步骤总数 | 72 |

---

## 全局开发约束

### G1. 禁止 Codex 自行变更的技术决策

- 包名：`muzero`；源码目录：`src/muzero/`；测试目录：`tests/`
- Python：`>=3.11,<3.12`；CLI：`typer`；配置校验：`pydantic>=2`
- YAML 解析：`pyyaml`；环境接口：`gymnasium`；训练框架：`torch`
- 日志：`torch.utils.tensorboard.SummaryWriter`
- Lint/format：`ruff`；类型检查：`pyright`；测试：`pytest`
- Ray：只放入 optional extra `ray`，默认安装不依赖 Ray
- MCTS v1：CPU 数组 tree + GPU batched inference
- `torch.compile`：默认关闭；AMP：使用 `torch.amp.autocast("cuda")`

### G2. 每个模块完成后的通用验证

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest -q
```

### G3. 文件路径约定

源码 → `src/muzero/`；测试 → `tests/unit/` 或 `tests/integration/`；配置 → `configs/`；脚本 → `scripts/`

---

# 模块 1-10：核心基线（已完成实现）

模块 1-10 实现了完整的 MuZero 基线：项目骨架、配置系统、核心类型、环境适配层、网络协议与 MLP 模型、TreeStorage/PUCT/搜索策略、标准 MCTS、ReplayBuffer/TargetBuilder、Trainer、CLI 与 smoke tests。

关键产出路径：
- `src/muzero/config/` — Pydantic v2 schema + YAML loader
- `src/muzero/core/` — types, specs, support transform, perspective, game history
- `src/muzero/envs/` — GameAdapter, GymnasiumAdapter, board games (TicTacToe, Connect4)
- `src/muzero/models/` — BaseMuZeroNetwork, MLPNetwork, EfficientZeroHeads, NetworkFactory
- `src/muzero/search/` — TreeStorage, puct_score, PUCTPolicy, DiscreteActionSampler, MCTS
- `src/muzero/replay/` — ReplayBuffer, PrioritizedReplayBuffer, TargetBuilder, ReanalyzeQueue
- `src/muzero/training/` — MuZeroLoss, OptimizerFactory, Trainer, CheckpointManager, WeightStore
- `src/muzero/cli/` — Typer CLI (train, benchmark, version)
- `src/muzero/execution/` — SelfPlayWorker

---

# 模块 11：InferenceBatcher 与 Batch MCTS v1（已完成实现）

## Step 11.1：实现 InferenceBatcher ✅
## Step 11.2：改造 MCTS 支持 batcher ✅
## Step 11.3：实现 BatchMCTS.run_batch ✅
## Step 11.4：接入 CLI benchmark ✅

---

# 模块 12：EfficientZero heads、loss、配置开关（已完成实现）

## Step 12.1：实现 EfficientZeroHeads ✅
## Step 12.2：扩展 MLPNetwork 输出 ✅
## Step 12.3：实现 EfficientZeroLoss ✅
## Step 12.4：接入 efficientzero config 与 smoke test ✅

---

# 模块 13：Sampled MuZero 与连续动作支持

## 概述
通过 ActionSampler 支持连续动作环境和 sampled search。
前置依赖：模块 12

## Step 13.1：实现 ContinuousActionSampler ✅
## Step 13.2：扩展 MLPNetwork 支持连续动作 dynamics ✅

## Step 13.3：修改 MCTS sampled mode

### 操作
修改 `MCTS._expand_root()` 与 `_expand_leaf()`：
- 从 `ActionSampleBatch.actions` 读取候选动作。
- 连续动作时 `actions` shape 为 `[B, K, action_dim]`，需适配存储。
- `build_policy_target()` 对连续动作返回 sampled target（actions + visit distribution）。
- 扩展 `SearchResult` 添加 `sampled_actions` 字段。

### 验证
```bash
uv run pytest tests/unit/test_mcts.py -q
```

## Step 13.4：Pendulum sampled smoke test

### 操作
创建 `tests/integration/test_pendulum_sampled_smoke.py`：
- load `configs/pendulum_sampled_muzero.yaml`
- build components with `ContinuousActionSampler`
- run one self-play episode
- verify action in action space range

### 验证
```bash
uv run pytest tests/integration/test_pendulum_sampled_smoke.py -q
```

---

# 模块 14：GumbelPolicy 低仿真搜索扩展

## 概述
实现 Gumbel root action sampling 与 sequential halving 占位策略。
前置依赖：模块 13

## Step 14.1：实现 GumbelPolicy 基础 ✅
## Step 14.2：接入 SearchPolicyFactory（通过显式选择策略类）
## Step 14.3：创建 Gumbel config ✅

## Step 14.4：Gumbel integration test

### 操作
创建 `tests/integration/test_gumbel_smoke.py`：
- load `configs/tictactoe_gumbel_muzero.yaml`
- build components with GumbelPolicy
- run one self-play episode
- assert actions in 0..8

### 验证
```bash
uv run pytest tests/integration/test_gumbel_smoke.py -q
```

---

# 模块 15：多人 value vector 完整验收

## 概述
确保 `num_players >= 1` 全链路可用。
前置依赖：模块 14

## Step 15.1-15.3：强化 TargetBuilder/TreeStorage/ThreePlayerToyEnv ✅

## Step 15.4：多人 integration test

### 操作
创建 `tests/integration/test_three_player_toy.py`：
- self-play episode 可运行
- reward vector shape 为 `[3]`
- network value shape 为 `[B, 3]`

### 验证
```bash
uv run pytest tests/integration/test_three_player_toy.py -q
```

---

# 模块 16：Checkpoint registry 与预训练权重接口

## 概述
实现 checkpoint metadata、inspect/export/import CLI。
前置依赖：模块 15

## Step 16.1：实现 CheckpointMetadata

### 操作
在 `src/muzero/training/checkpoint.py` 添加：

```python
class CheckpointMetadata(BaseModel):
    format_version: int = 1
    project_name: str
    algorithm: Literal["muzero", "efficientzero", "sampled_muzero", "gumbel_muzero"]
    env_id: str
    network_type: Literal["mlp", "residual", "conv"]
    observation_shape: list[int]
    action_space: dict[str, Any]
    num_players: int
    training_steps: int
    created_at: str
    git_commit: str | None = None
    config_hash: str
```

实现 `build_checkpoint_metadata(config, env_spec, training_steps) -> CheckpointMetadata` 和 `compute_config_hash(config) -> str`。

### 验证
```bash
uv run pytest tests/unit/test_checkpoint_metadata.py -q
```

## Step 16.2：实现 checkpoint inspect/export/import

扩展 `CheckpointManager`：
- `inspect(path) -> CheckpointMetadata`
- `export(path, out_dir) -> None`（输出 model.pt, config.yaml, metadata.yaml, README.md）
- `import_checkpoint(path) -> CheckpointState`

## Step 16.3：实现 checkpoint CLI

Typer 命令：`muzero checkpoint inspect|export|import`

## Step 16.4：checkpoint roundtrip integration test

创建 `tests/integration/test_checkpoint_roundtrip.py`

---

# 模块 17：Ray optional backend

## 概述
实现 Ray 可选执行后端，算法模块不依赖 Ray。
前置依赖：模块 16

## Step 17.1：实现 ExecutorBackend 与 LocalExecutorBackend
## Step 17.2：实现 EvaluatorWorker 与 ReanalyzeWorker 占位
## Step 17.3：实现 RayExecutorBackend
## Step 17.4：接入 backend factory 和 Ray smoke

关键约束：Ray 只允许在 `src/muzero/execution/ray_backend.py` import。

---

# 模块 18：Benchmark、日志、最终工程验收

## 概述
完善日志、性能指标、benchmark、README，完成全项目验收。
前置依赖：模块 17

## Step 18.1：实现 MetricsLogger 和 PerformanceTracker

```python
class MetricsLogger:
    def __init__(self, log_dir, enable_tensorboard=True): ...
    def log_scalar(name, value, step): ...
    def close(): ...

class PerformanceTracker:
    def record_inference_latency(ms, batch_size): ...
    def summary() -> PerformanceSummary: ...
```

## Step 18.2：接入 Trainer 与 SelfPlay 日志
## Step 18.3：完善 benchmark runner（replay, training 组件）
## Step 18.4：最终文档与全量验收

---

# 总体验收标准

- [ ] 标准 MuZero baseline 可运行 CartPole
- [ ] `uv sync --extra dev` 成功
- [ ] `uv run ruff format --check .` 成功
- [ ] `uv run ruff check .` 成功
- [ ] `uv run pyright` 成功
- [ ] `uv run pytest -q` 成功
- [ ] `uv run muzero --help` 成功
- [ ] `uv run muzero train --config configs/cartpole_muzero.yaml --profile cpu_debug --seed 0` 成功
