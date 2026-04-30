# Gumbel MuZero — 实现参考

## 来源
- 论文/项目：Policy improvement by planning with Gumbel
- 核心贡献：用 Gumbel sampling 和 sequential halving 改善少 simulation 条件下的 policy improvement。

## 关键实现要点
- `GumbelPolicy` 必须实现 `SearchPolicy` 接口。
- root action sampling 使用 `policy_logits + gumbel_noise` 后取 top-k。
- 非 root selection 第一版复用 `PUCTPolicy`。
- `sequential_halving()` 第一版返回 sampled root actions 中 visit count 最大动作。
- 不复制整套 MCTS；只替换 SearchPolicy。

## 对应模块
- 使用方：plan.md 中的模块 14
- 集成方式：通过 `SearchPolicyFactory` 在 `algorithm.name == "gumbel_muzero"` 时注入 `GumbelPolicy`。

## 注意事项
- legal action mask 必须在 root sampling 中生效。
- 第一版不要求完整 sequential halving，但接口必须保留。
- GumbelPolicy 不能破坏标准 PUCTPolicy 测试。
