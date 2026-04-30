# 技术调研报告

## 0. 调研结论

本项目建议采用 **“从零重构的模块化 MuZero Core + EfficientZero 样本效率增强 + Sampled/Gumbel 搜索扩展 + 可选 Ray 分布式后端”** 作为主路线。

针对用户确认的硬件环境 **RTX 4060 Laptop**，默认设计应优先支持 **单机单 GPU、8GB 显存约束、CPU 自博弈 + GPU 批量推理队列**。Ray 不作为基础依赖强制引入，而是设计成可选执行后端；这样能同时满足本机开发、后续多进程、多 GPU、集群扩展。

核心推荐：
- **基础算法**：标准 MuZero，作为可验证基线。
- **样本效率**：引入 EfficientZero 的一致性损失、value prefix、off-policy/reanalyze 思路。
- **连续动作空间**：采用 Sampled MuZero / EfficientZero V2 风格的 action sampler + sampled search。
- **少仿真高效率搜索**：预留 Gumbel MuZero 的 root action sampling / sequential halving 机制。
- **Batch MCTS**：第一版实现“CPU 树结构 + GPU batched inference queue”；第二版再做更激进的 tensorized/GPU tree。
- **多人游戏**：从数据结构层面支持 `num_players >= 1`，value 默认采用 `value_vector[num_players]`，二人零和游戏可通过 perspective projection 得到标量值。
- **预训练权重**：实现 checkpoint registry、metadata schema、导入导出 CLI；先支持本地产物，不承诺托管公共权重。
- **工程栈**：Python 3.11 + PyTorch 2.x + Gymnasium + TensorBoard；Ray optional；配置使用 YAML + dataclass/Pydantic 校验。

---

## 1. 调研范围

### 1.1 用户输入

| 项 | 决策 |
|---|---|
| 项目方式 | 从零重构 |
| 覆盖范围 | 全部覆盖：Batch MCTS、性能优化、连续动作空间、超过两人游戏、预训练权重、工程化 |
| 是否上传原代码 | 不上传，直接重构 |
| 运行硬件 | RTX 4060 Laptop |
| Python / PyTorch / Gymnasium | 确认按 Python 3.11 + PyTorch 2.x + Gymnasium 规划 |
| 指标排序 | 由本报告推荐 |
| Ray 策略 | 由本报告推荐 |

### 1.2 调研关键词

- MuZero
- EfficientZero
- EfficientZero V2
- Sampled MuZero
- Gumbel MuZero
- Batch MCTS
- Batched inference for MCTS
- Multi-player reinforcement learning
- Continuous action space planning
- Gymnasium environment API
- PyTorch 2.x reinforcement learning
- Ray actors / tasks for RL self-play
- RTX 4060 Laptop GPU constraints

### 1.3 覆盖时间范围

- 核心算法：2019-2026
- 工程依赖：以 2026-04-29 可用资料为准
- 当前阶段目标：技术路线选择，不进入文件级架构设计；架构文件将在阶段 2 生成

---

## 2. 关键事实与技术背景

### 2.1 原项目定位

`PROJECT_OVERVIEW.md` 将项目定义为基于 Google DeepMind MuZero 论文的通用强化学习算法实现，主要依赖为 Python、PyTorch、Ray、TensorBoard、Gym，并已包含 CartPole、LunarLander、Tic-tac-toe、Connect4、Gomoku、Breakout、Gridworld 等环境方向。

原项目进一步列出以下改进方向：
- Batch MCTS
- 支持超过两人的游戏
- 更多预训练权重
- 性能优化

这与本轮“从零重构、全部覆盖”的目标一致，但本轮不做旧仓库增量修改，而是将这些方向纳入新架构的第一性设计。

### 2.2 MuZero 核心

MuZero 的核心思想是用 learned model 支持 tree-based search；模型不直接学习完整环境动力学，而是学习规划需要的 reward、policy、value。三类网络为：

- `representation_network(observation) -> hidden_state`
- `dynamics_network(hidden_state, action) -> next_hidden_state, reward`
- `prediction_network(hidden_state) -> policy_logits, value`

调研结论：MuZero 必须作为第一阶段实现基线。所有 EfficientZero、Sampled MuZero、Gumbel MuZero、Batch MCTS 都应作为 MuZero Core 的扩展，而不是替代基础 MuZero。

---

## 3. 技术方案对比

| 方案 | 来源 | 核心原理 | 优势 | 劣势 | 适用场景 | 可行性评分 |
|---|---|---|---|---|---|---|
| 标准 MuZero | DeepMind MuZero 2019 | 学习 representation/dynamics/prediction，用 MCTS 生成 policy/value target | 原理清晰；适合离散动作、棋类、Atari；是所有扩展的基线 | 样本效率一般；MCTS 计算重；连续动作空间需改造 | 第一版基线、离散动作环境、棋类、经典控制离散化 | 5/5 |
| EfficientZero | NeurIPS 2021 | 在 MuZero 上加入一致性损失、value prefix、off-policy/reanalyze 改进 | 显著提高样本效率；适合有限交互预算；能改善模型学习 | 实现复杂度高于 MuZero；训练目标更多；调参难度增加 | Atari 100k、低样本预算、本机训练 | 4/5 |
| EfficientZero V2 | 2024 | 将 EfficientZero 扩展到离散/连续动作、视觉/低维输入等多域任务 | 方向最符合“全覆盖”；连续控制与通用样本效率均被纳入 | 论文实现复杂；完整复刻不适合作为第一版 | 第二阶段增强、连续控制、统一算法框架 | 4/5 |
| Sampled MuZero | ICML 2021 | 在复杂/连续动作空间中只对采样动作子集做 policy evaluation/improvement | 解决动作不可枚举问题；适合连续控制 | 需要 action sampler；policy target 不再是全动作分布；训练数据结构更复杂 | 连续动作空间、高维动作空间 | 4/5 |
| Gumbel MuZero | ICLR 2022 | 使用 Gumbel sampling + sequential halving 改善少仿真下的 policy improvement | 减少 MCTS simulations 预算；适合 RTX 4060 laptop 的算力限制 | 实现细节容易出错；需独立验证少仿真收益 | 低仿真预算、棋类、离散动作 | 4/5 |
| Batch MCTS：CPU tree + GPU inference queue | 工程方案 | tree expansion/selection 在 CPU，网络推理按队列批量送 GPU | 实现可控；适配 PyTorch；显存风险较低；适合本机 4060 | 不是全 GPU tree，CPU 仍可能成为瓶颈 | 第一版性能优化主线 | 5/5 |
| Batch MCTS：Tensorized/GPU tree | MCTX/JAX 思路 | tree 节点和搜索过程张量化，支持 JIT/batch 并行 | 理论吞吐高；适合 TPU/GPU | PyTorch 下实现难；调试成本高；对 8GB 显存不友好 | 第二版或实验模块 | 2.5/5 |
| Ray 强制分布式 | Ray | actor/task 组织 self-play、trainer、evaluator | 易扩展到多进程/集群；适合大规模自博弈 | 对单机 laptop 增加依赖和调试复杂度 | 集群训练、多 GPU | 3/5 |
| Ray optional backend | Ray + local backend abstraction | 默认 local，按配置切换 Ray actors | 保持本机简单，同时保留扩展性 | 架构需提前抽象 executor 接口 | 推荐主线 | 5/5 |
| LightZero 作为参考 | OpenDILab | 已实现多种 MCTS+RL 算法 | 参考模块拆分与算法接口 | 不能直接复制；复杂度较高 | 参考 API/测试/算法组织 | 4/5 |
| MCTX 作为参考 | Google DeepMind | JAX-native batched MCTS，支持 AlphaZero/MuZero/Gumbel MuZero | 对 batch MCTS 设计有参考价值 | 技术栈是 JAX，不符合本项目 PyTorch 主线 | 参考 tree data layout / batch policy | 3.5/5 |

---

## 4. 推荐方案

### 4.1 首选方案

采用：

> **Modular MuZero Core + EfficientZero-style sample efficiency + Sampled/Gumbel search extensions + optional Ray backend**

该方案分成基础可跑、样本效率、动作空间泛化、性能扩展四层，不把所有论文特性一次性塞进最小可运行版本。

#### 推荐原因

1. **适合 RTX 4060 Laptop**
   - 该 GPU 通常为 8GB 显存级别，不适合默认使用巨型网络、超大 replay buffer、全 GPU tree search。
   - 第一版应优先做小 batch、AMP、gradient accumulation、batched inference queue，而不是追求极限分布式。

2. **降低算法实现风险**
   - 标准 MuZero 先跑通离散动作与棋类。
   - EfficientZero、Sampled MuZero、Gumbel MuZero 作为扩展策略逐步接入。

3. **满足“全部覆盖”但不牺牲可验证性**
   - Batch MCTS、连续动作、多人游戏、预训练权重、性能优化都进入架构。
   - 每个方向都有独立接口、单元测试、最小验收环境。

4. **Ray 可选比 Ray 必选更合理**
   - 单机开发时直接运行 local executor。
   - 需要并行 self-play 时再启用 Ray executor。
   - 阶段 2 架构中应强制规定：业务模块不得直接依赖 `ray`，只能依赖 `ExecutorBackend` 抽象。

### 4.2 备选方案

#### 备选 A：只重构标准 MuZero

不推荐。虽然实现简单，但无法覆盖用户要求的连续动作、Batch MCTS、性能优化、多人游戏与预训练权重目标。

#### 备选 B：直接复刻 EfficientZero V2 全功能

不推荐作为第一版。EfficientZero V2 覆盖范围强，但完整复刻会导致架构和调试复杂度过高，不利于 Codex 连续执行。

#### 备选 C：JAX + MCTX 重写

不推荐。本项目已确认 Python + PyTorch 方向，且原项目生态也以 PyTorch/Ray 为核心。MCTX 适合作为 batched tree design 参考，不适合作为主技术栈。

---

## 5. 推荐指标排序

针对“从零重构 + RTX 4060 Laptop + 全部覆盖”的组合，推荐指标优先级如下：

| 排名 | 指标 | 推荐权重 | 原因 |
|---:|---|---:|---|
| 1 | 正确性与可复现性 | 25% | MuZero 类算法容易“能跑但不对”；必须优先保证 target、backup、player perspective、replay sampling 正确 |
| 2 | 可维护性与可扩展性 | 20% | 本项目要覆盖多算法、多环境、多动作空间，接口设计比短期速度更重要 |
| 3 | 样本效率 | 20% | 本机训练预算有限，EfficientZero 系改进应优先纳入 |
| 4 | 单 GPU 吞吐与训练速度 | 15% | RTX 4060 laptop 需要重视 batched inference、AMP、compile、DataLoader/queue |
| 5 | 最终回报/胜率 | 10% | 重要但不应牺牲架构正确性；可在算法稳定后提升 |
| 6 | 显存占用 | 5% | 通过默认配置控制；不作为牺牲算法正确性的主目标 |
| 7 | 集群扩展能力 | 5% | Ray optional 保留扩展，但不作为本机第一版主目标 |

---

## 6. 推荐工程技术栈

### 6.1 基础环境

| 类别 | 推荐 |
|---|---|
| Python | 3.11 |
| 深度学习框架 | PyTorch 2.x stable |
| CUDA | 使用与本机驱动兼容的官方 PyTorch CUDA wheel |
| 环境 API | Gymnasium |
| 配置 | YAML + dataclass/Pydantic schema validation |
| CLI | Typer 或 argparse；阶段 2 决定具体包 |
| 日志 | Python logging + TensorBoard |
| 实验记录 | TensorBoard 必选；Weights & Biases 可选 |
| 测试 | pytest |
| 类型检查 | pyright 或 mypy；阶段 2 固定一个 |
| 格式化 | ruff format + ruff check |
| 分布式 | 默认 local executor；Ray optional |
| 打包 | `pyproject.toml`，推荐 `uv` 管理依赖 |

### 6.2 RTX 4060 Laptop 默认运行策略

默认 profile 名称建议：

```yaml
profile: laptop_rtx4060_8gb
```

默认约束建议：

```yaml
device: cuda
precision: amp_bf16_or_fp16
num_self_play_workers: 2
num_envs_per_worker: 1
inference_batch_size: 16
training_batch_size: 64
unroll_steps: 5
td_steps: 10
num_simulations: 25
replay_buffer_size: 50000
checkpoint_interval_steps: 1000
```

说明：
- 这些值不是最终超参数，而是阶段 2/3 的默认配置起点。
- Atari/视觉任务应使用更小 batch 或更小网络。
- 棋类任务可提高 `num_simulations`。
- `torch.compile` 设为 opt-in，不能默认强制启用，因为动态控制流和 MCTS 组合容易触发 graph break。

---

## 7. 关键参考资料

### 7.1 MuZero

| 项 | 内容 |
|---|---|
| 标题 | Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model |
| 作者 | Julian Schrittwieser et al. |
| 年份 | 2019/2020 |
| 来源 | arXiv / Nature |
| URL | https://arxiv.org/abs/1911.08265 |
| 核心贡献 | 用 learned model + tree search，在不了解环境动力学的情况下学习 reward、policy、value，用于规划 |
| 对本项目作用 | 基础算法核心；所有接口必须先满足 MuZero Core |

### 7.2 EfficientZero

| 项 | 内容 |
|---|---|
| 标题 | Mastering Atari Games with Limited Data |
| 作者 | Weirui Ye et al. |
| 年份 | 2021 |
| 来源 | NeurIPS 2021 |
| URL | https://arxiv.org/abs/2111.00210 |
| 核心贡献 | 基于 MuZero 改进样本效率，引入 temporally consistent model learning、value prefix、off-policy correction/reanalyze 思路 |
| 对本项目作用 | 样本效率增强模块；适合本机有限计算预算 |

### 7.3 EfficientZero V2

| 项 | 内容 |
|---|---|
| 标题 | EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data |
| 作者 | Shengjie Wang et al. |
| 年份 | 2024 |
| 来源 | ICML 2024 |
| URL | https://arxiv.org/abs/2403.00564 |
| 核心贡献 | 将 EfficientZero 扩展到离散/连续动作、视觉/低维输入等多领域 |
| 对本项目作用 | 连续动作与统一算法框架参考；不建议第一版完整复刻 |

### 7.4 Sampled MuZero

| 项 | 内容 |
|---|---|
| 标题 | Learning and Planning in Complex Action Spaces |
| 作者 | Thomas Hubert et al. |
| 年份 | 2021 |
| 来源 | ICML 2021 |
| URL | https://arxiv.org/abs/2104.06303 |
| 核心贡献 | 对复杂/连续动作空间只采样动作子集并进行 policy evaluation/improvement |
| 对本项目作用 | 连续动作空间设计的直接理论来源 |

### 7.5 Gumbel MuZero

| 项 | 内容 |
|---|---|
| 标题 | Policy improvement by planning with Gumbel |
| 作者 | Ivo Danihelka et al. |
| 年份 | 2022 |
| 来源 | ICLR 2022 / OpenReview |
| URL | https://openreview.net/forum?id=bERaNdoegnO |
| 核心贡献 | 用 Gumbel sampling 与 policy improvement 机制改善少仿真预算下的搜索 |
| 对本项目作用 | 低 simulation 预算的搜索优化；适合 4060 laptop profile |

### 7.6 MCTX

| 项 | 内容 |
|---|---|
| 项目 | google-deepmind/mctx |
| URL | https://github.com/google-deepmind/mctx |
| 核心贡献 | JAX-native MCTS，支持 AlphaZero、MuZero、Gumbel MuZero，搜索算法按 batch 并行 |
| 对本项目作用 | Batch MCTS 数据结构与 batch API 参考；不作为主依赖 |

### 7.7 LightZero

| 项 | 内容 |
|---|---|
| 项目 | OpenDILab LightZero |
| URL | https://github.com/opendilab/LightZero |
| 核心贡献 | 统一 MCTS+RL 工具包，支持 MuZero、EfficientZero、Gumbel MuZero 等 |
| 对本项目作用 | 模块拆分、训练入口、评估入口、配置组织参考；不直接复制实现 |

### 7.8 Gymnasium

| 项 | 内容 |
|---|---|
| 项目 | Gymnasium |
| URL | https://gymnasium.farama.org/ |
| 核心贡献 | OpenAI Gym 的维护分支，提供标准 RL 环境 API |
| 对本项目作用 | 替代旧 Gym；阶段 2 中环境适配层必须使用 Gymnasium Step API：`terminated` / `truncated` |

### 7.9 Ray

| 项 | 内容 |
|---|---|
| 项目 | Ray |
| URL | https://docs.ray.io/ |
| 核心贡献 | 面向 Python/AI 应用的分布式任务与 actor 框架 |
| 对本项目作用 | 可选 self-play / evaluator / reanalyze backend；不得作为 MuZero Core 的强依赖 |

### 7.10 PyTorch

| 项 | 内容 |
|---|---|
| 项目 | PyTorch |
| URL | https://pytorch.org/get-started/locally/ |
| 核心贡献 | 深度学习框架，提供 GPU/CPU tensor、autograd、AMP、compile 等能力 |
| 对本项目作用 | 主训练框架；模型、loss、checkpoint、AMP 均基于 PyTorch |

---

## 8. 需要蒸馏给 Codex 的技术要点

> 本节不是最终开发计划，但其中的实现约束将在阶段 2/3 转化为具体文件、类、方法、步骤和验证命令。

### 8.1 MuZero Core 必须抽象为算法无关接口

必须拆成以下概念，不允许写成单个 `muzero.py` 巨型脚本：

```python
representation(observation) -> hidden_state
dynamics(hidden_state, action) -> DynamicsOutput(next_hidden_state, reward_or_value_prefix)
prediction(hidden_state) -> PredictionOutput(policy_logits, value)
```

所有网络必须实现统一协议：

```python
class MuZeroNetworkProtocol(Protocol):
    def initial_inference(self, observation_batch: Tensor) -> NetworkOutput: ...
    def recurrent_inference(self, hidden_state_batch: Tensor, action_batch: Tensor) -> NetworkOutput: ...
```

其中：

```python
NetworkOutput:
    value: Tensor              # [B] 或 [B, num_players]
    reward: Tensor             # [B] 或 support logits
    policy_logits: Tensor      # [B, action_dim] 或 sampled action logits
    hidden_state: Tensor       # [B, ...]
```

### 8.2 MCTS 必须支持 batch inference

MCTS 第一版不做全 GPU tree，但必须让网络推理批量化：

```python
class InferenceBatcher:
    def enqueue_initial(observation, callback): ...
    def enqueue_recurrent(hidden_state, action, callback): ...
    def flush() -> None: ...
```

搜索 worker 执行 selection/expansion 时不能每个 node 单独调用 GPU；必须进入 batcher，由 batcher 聚合为 tensor batch 后调用网络。

### 8.3 Tree 节点不要用递归对象树作为唯一实现

可以提供 debug-friendly object tree，但正式 MCTS 存储应使用数组结构：

```python
visit_count: np.ndarray[int32]
value_sum: np.ndarray[float32]
prior: np.ndarray[float32]
reward: np.ndarray[float32]
parent_index: np.ndarray[int32]
first_child_index: np.ndarray[int32]
num_children: np.ndarray[int32]
action_from_parent: np.ndarray
hidden_state_id: np.ndarray[int32]
to_play: np.ndarray[int32]
```

原因：
- 更容易 batch 化；
- 更容易 checkpoint/debug；
- 更接近未来 GPU/tensorized tree；
- 避免 Python 对象嵌套导致性能不可控。

### 8.4 PUCT / UCB 选择公式必须集中实现

搜索选择逻辑必须集中到一个函数，避免多个算法复制：

```python
def puct_score(
    parent_visit_count: int,
    child_visit_count: int,
    child_prior: float,
    child_value: float,
    pb_c_base: float,
    pb_c_init: float,
) -> float:
    pb_c = math.log((parent_visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= math.sqrt(parent_visit_count) / (child_visit_count + 1)
    return child_value + pb_c * child_prior
```

Gumbel MuZero 不能改写基础 PUCT 文件，而应通过 `SearchPolicy` 策略类替换 root action selection。

### 8.5 EfficientZero 扩展必须模块化

EfficientZero 相关训练目标拆成独立 loss 模块：

```python
class MuZeroLoss:
    policy_loss(...)
    value_loss(...)
    reward_loss(...)

class EfficientZeroLoss:
    consistency_loss(...)
    value_prefix_loss(...)
    off_policy_correction_loss_or_target(...)
```

第一版实现顺序：
1. 标准 MuZero loss。
2. value prefix head，替代/扩展 reward head。
3. consistency projection head。
4. reanalyze/off-policy correction。

不得在第一版训练循环中硬编码所有 loss；必须通过配置启用：

```yaml
algorithm:
  name: efficientzero
  use_consistency_loss: true
  use_value_prefix: true
  use_reanalyze: false
```

### 8.6 连续动作空间必须通过 ActionSampler 接口进入 MCTS

不要把连续动作离散化写死在环境里。必须引入：

```python
class ActionSampler(Protocol):
    def sample(
        self,
        policy_output: Tensor,
        observation_or_hidden_state: Tensor,
        num_samples: int,
        legal_action_mask: Tensor | None,
    ) -> ActionSampleBatch: ...
```

`ActionSampleBatch` 至少包含：

```python
actions: Tensor              # [B, K, action_dim] 或 [B, K]
log_probs: Tensor | None     # [B, K]
priors: Tensor               # [B, K]
```

Sampled MuZero 搜索只在 `K` 个候选动作上展开，policy target 也只对候选动作集合定义。

### 8.7 多人游戏默认使用 value vector

对于 `num_players > 2`，不要只使用当前玩家标量 value。推荐：

```python
value: Tensor  # [B, num_players]
reward: Tensor # [B, num_players] 或 [B] 由环境声明
```

二人零和游戏可通过：

```python
scalar_value_for_player = value_vector[current_player] - mean(value_vector[other_players])
```

或者使用环境提供的 `PerspectiveTransform`。但底层 replay 和 network output 应保留 vector 能力，避免后续多人游戏重构。

### 8.8 Game / Env API 必须区分 Gymnasium 单智能体与通用博弈

建议设计两层：

```python
class GameAdapter(Protocol):
    def reset(seed: int | None = None) -> TimeStep: ...
    def step(action: Action) -> TimeStep: ...
    def legal_actions() -> LegalActions: ...
    def current_player() -> int: ...
    def num_players() -> int: ...

class GymnasiumAdapter(GameAdapter):
    ...
```

Gymnasium 的 `terminated` 与 `truncated` 必须保留，不允许重新压成旧版 `done` 后丢失信息。

### 8.9 Replay Buffer 必须从第一版支持优先级与 reanalyze 预留字段

最小数据结构：

```python
GameHistory:
    observations: list[Observation]
    actions: list[Action]
    rewards: list[Reward]
    players: list[int]
    root_values: list[Value]
    child_visit_distributions: list[PolicyTarget]
    legal_action_masks: list[Mask | None]
    search_metadata: list[SearchMetadata]
```

Replay sample 返回：

```python
TrainingBatch:
    observation: Tensor
    action_unroll: Tensor
    target_value: Tensor
    target_reward_or_value_prefix: Tensor
    target_policy: Tensor
    target_mask: Tensor
    importance_weights: Tensor | None
    indices: Tensor | None
```

### 8.10 Pretrained weights 应定义元数据 schema

checkpoint 文件必须包含：

```yaml
format_version: 1
project_name: muzero-rebuilt
algorithm: muzero | efficientzero | sampled_muzero | gumbel_muzero
env_id: CartPole-v1
network_type: mlp | residual | conv
observation_shape: [...]
action_space:
  type: discrete | continuous | sampled
  n: ...
num_players: 1
training_steps: 0
created_at: ISO-8601
git_commit: optional
config_hash: sha256
```

CLI 必须至少预留：

```bash
muzero checkpoint inspect path/to/checkpoint.pt
muzero checkpoint export path/to/checkpoint.pt --out exported/
muzero checkpoint import path/to/checkpoint.pt
```

### 8.11 性能优化策略

第一版必须做：
- AMP 训练：`torch.cuda.amp.autocast` / `GradScaler` 视 PyTorch 版本确定。
- 网络推理 batch 化。
- Replay sample 使用 numpy/tensor 批量拼接，避免 Python 循环逐项转 tensor。
- 训练循环中观测预处理只做一次。
- 配置中明确 `device`, `precision`, `batch_size`, `num_simulations`。
- TensorBoard 记录每秒 self-play step、每秒 training step、GPU memory allocated、batch inference latency。

第二版再做：
- `torch.compile` opt-in。
- tensorized tree。
- Ray actor pool。
- prioritized replay 优化。
- reanalyze worker。
- 多 GPU trainer / self-play 分离。

---

## 9. 推荐阶段拆分

### 9.1 阶段 2 架构设计应覆盖的模块

建议阶段 2 设计以下模块：

1. `muzero.config`
   - 配置 dataclass / schema / YAML loader

2. `muzero.envs`
   - `GameAdapter`
   - `GymnasiumAdapter`
   - board game adapter
   - continuous control adapter

3. `muzero.core`
   - `types.py`
   - `game_history.py`
   - `support.py`
   - `player.py`

4. `muzero.models`
   - `MuZeroNetworkProtocol`
   - `MLPNetwork`
   - `ResidualNetwork`
   - `ConvNetwork`
   - `EfficientZeroHeads`

5. `muzero.search`
   - `MCTS`
   - `BatchMCTS`
   - `SearchPolicy`
   - `PUCTPolicy`
   - `GumbelPolicy`
   - `ActionSampler`

6. `muzero.replay`
   - replay buffer
   - prioritized replay
   - target builder
   - reanalyze placeholder

7. `muzero.training`
   - trainer
   - loss modules
   - optimizer builder
   - checkpoint manager

8. `muzero.execution`
   - local executor
   - optional Ray executor
   - inference batcher
   - self-play worker
   - evaluator worker

9. `muzero.cli`
   - train
   - eval
   - self-play
   - benchmark
   - checkpoint inspect/export/import

10. `tests`
   - unit tests for support transform, MCTS backup, target builder
   - integration tests for CartPole / TicTacToe smoke training
   - deterministic seed tests

### 9.2 阶段 3 计划制定应采用的实现顺序

推荐拓扑顺序：

1. 项目骨架与依赖
2. 配置系统
3. 核心类型与 support transform
4. 环境适配层
5. 网络协议与最小 MLP 模型
6. 标准 MCTS
7. Replay buffer 与 target builder
8. 标准 MuZero trainer
9. CLI + smoke tests
10. Batch inference queue
11. EfficientZero loss 扩展
12. Sampled action support
13. Gumbel policy extension
14. 多人 value vector
15. Checkpoint registry / pretrained interface
16. Ray optional executor
17. Benchmark 与性能 profiling

---

## 10. 风险与应对

| 风险 | 影响 | 应对策略 |
|---|---|---|
| 一次性实现全部论文特性导致不可调试 | 项目失败概率高 | 阶段 3 计划必须先做标准 MuZero baseline，再逐步启用扩展 |
| RTX 4060 Laptop 显存不足 | 视觉任务训练 OOM | 默认小网络、小 batch、AMP、profile 配置；Atari 作为后续验收，不作为第一个 smoke test |
| Ray 引入过早 | 本机开发复杂度升高 | Ray optional；local executor 必须完整可用 |
| 连续动作与离散动作混写 | 架构失控 | 统一 `ActionSpaceSpec` + `ActionSampler` |
| 多人游戏后补困难 | 数据结构重构成本高 | 第一版 replay/network output 就支持 `num_players` 和 value vector |
| Batch MCTS 性能不达预期 | 搜索吞吐低 | 第一版先保证 batched inference；第二版再 tensorize tree |
| EfficientZero 训练目标复杂 | loss bug 难定位 | 每个 loss 单独测试；配置开关逐项启用 |
| 预训练权重兼容性差 | 权重不可复用 | checkpoint metadata schema 从第一版定义 |
| Gym/Gymnasium API 混淆 | truncated/terminated 错误影响 bootstrapping | 只支持 Gymnasium API；旧 Gym 通过 adapter 迁移 |

---

## 11. 建议下载/保留的论文与资料

必须保留：
1. MuZero — https://arxiv.org/abs/1911.08265
2. EfficientZero — https://arxiv.org/abs/2111.00210
3. Sampled MuZero — https://arxiv.org/abs/2104.06303
4. Gumbel MuZero — https://openreview.net/forum?id=bERaNdoegnO
5. EfficientZero V2 — https://arxiv.org/abs/2403.00564

工程参考：
1. MCTX — https://github.com/google-deepmind/mctx
2. LightZero — https://github.com/opendilab/LightZero
3. Gymnasium docs — https://gymnasium.farama.org/
4. Ray docs — https://docs.ray.io/
5. PyTorch local install docs — https://pytorch.org/get-started/locally/

---

## 12. 阶段 1 结论检查

### 12.1 推荐技术路线

确认推荐：

```text
Python 3.11
PyTorch 2.x
Gymnasium
TensorBoard
Local executor by default
Ray optional backend
Standard MuZero baseline
EfficientZero sample-efficiency extensions
Sampled MuZero / EfficientZero V2 continuous-action extension
Gumbel MuZero low-simulation search extension
CPU tree + GPU batched inference as Batch MCTS v1
Tensorized/GPU tree as later experimental v2
```

### 12.2 是否满足用户“全部覆盖”

| 用户目标 | 是否覆盖 | 调研结论 |
|---|---|---|
| Batch MCTS | 是 | v1 用 CPU tree + GPU batched inference；v2 tensorized tree |
| 性能优化 | 是 | AMP、batched inference、profile、Ray optional、torch.compile opt-in |
| 连续动作空间 | 是 | ActionSampler + Sampled MuZero / EZ-V2 思路 |
| 超过两人游戏 | 是 | value vector + current player + generalized game adapter |
| 预训练权重 | 是 | checkpoint metadata schema + registry + import/export CLI |
| 工程化 | 是 | pyproject、config schema、tests、CLI、logging、TensorBoard、CI 预留 |

### 12.3 进入阶段 2 的输入

阶段 2 架构设计将默认采用：
- 部署环境：本机 RTX 4060 Laptop，单 GPU 优先，后续可扩展 Ray。
- 项目方式：从零重构。
- 主技术栈：Python 3.11 + PyTorch 2.x + Gymnasium。
- 分布式策略：local executor 默认，Ray optional。
- 算法路线：MuZero Core first，EfficientZero/Sampled/Gumbel/Batch extensions second。
- 数据规模：默认本机训练规模，Atari/视觉任务作为可选 profile。
- 外部 API：无强制外部 API，不依赖云服务。

---

## 13. 参考来源

- MuZero: https://arxiv.org/abs/1911.08265
- EfficientZero: https://arxiv.org/abs/2111.00210
- Sampled MuZero: https://arxiv.org/abs/2104.06303
- Gumbel MuZero: https://openreview.net/forum?id=bERaNdoegnO
- EfficientZero V2: https://arxiv.org/abs/2403.00564
- Gymnasium: https://gymnasium.farama.org/
- Ray docs: https://docs.ray.io/
- PyTorch install docs: https://pytorch.org/get-started/locally/
- MCTX: https://github.com/google-deepmind/mctx
- LightZero: https://github.com/opendilab/LightZero
- NVIDIA RTX 4060 Laptop GPU reference: https://www.notebookcheck.net/NVIDIA-GeForce-RTX-4060-Laptop-GPU-Benchmarks-and-Specs.675692.0.html
