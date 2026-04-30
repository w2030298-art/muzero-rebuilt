# Sampled MuZero 与连续动作 — 实现参考

## 来源
- 论文/项目：Sampled MuZero / EfficientZero V2
- 核心贡献：在复杂或连续动作空间中只对候选采样动作集合进行搜索和策略改进。

## 关键实现要点
- 连续动作必须通过 `ActionSampler` 进入 MCTS。
- 环境层只暴露 action space，不负责硬编码离散化。
- `ContinuousActionSampler` 第一版将 policy output 解释为 action mean。
- 采样动作从 Normal(mean, sampling_std) 生成，并 clamp 到 action bounds。
- `ActionSampleBatch.actions` shape 为 `[B, K, action_dim]`。
- `ActionSampleBatch.priors` shape 为 `[B, K]`。
- 连续动作 SearchResult 需要保存 `sampled_actions`。

## 对应模块
- 使用方：plan.md 中的模块 13
- 集成方式：扩展 MLP dynamics action encoder，修改 MCTS expansion 和 TargetBuilder sampled policy target。

## 注意事项
- 不要把 Pendulum 动作空间手工切成固定离散 bins。
- 连续动作 policy loss 第一版可以降级为 smoke 可用方案，但必须在配置中标注 sampled mode。
- 优先保证 action bounds、shape、self-play smoke 正确。
