# MuZero Core — 实现参考

## 来源
- 论文/项目：MuZero: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model
- 核心贡献：用 learned model 预测 reward、policy、value，并结合 tree search 做规划。

## 关键实现要点
- 网络必须拆为 representation、dynamics、prediction 三个逻辑部分。
- 搜索层只能调用 `initial_inference(observation_batch)` 与 `recurrent_inference(hidden_state_batch, action_batch)`。
- `NetworkOutput` 必须包含 `value`、`reward`、`policy_logits`、`hidden_state`。
- 标准训练 target 包含 value target、reward target、policy target。
- MCTS 生成的 visit distribution 是 policy target，不直接使用网络 policy 作为最终训练 target。

## 对应模块
- 使用方：plan.md 中的模块 5、6、7、8、9、10
- 集成方式：先完成 MLP baseline、TreeStorage、PUCT、标准 MCTS、ReplayBuffer、Trainer，再接 CLI smoke train。

## 注意事项
- 不要把 `muzero.py` 写成单文件巨型脚本。
- Trainer 不直接访问环境。
- TargetBuilder 不写在 Trainer 内部。
- MCTS 不绕过网络协议直接调用 representation/dynamics/prediction。
