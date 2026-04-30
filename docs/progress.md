# 开发进度

## 当前状态
- 当前阶段：模块 1-9 核心完成（40/72 Steps），模块 10 部分完成，模块 11-18 待实现
- 测试总数：100 passed
- 最后更新：2026-04-29
- 状态：进行中

## 模块进度

### 模块 1：项目骨架与工具链
- [x] Step 1.1: 创建基础目录与包入口 ✅ 2026-04-29
- [x] Step 1.2: 配置 `pyproject.toml` ✅ 2026-04-29
- [x] Step 1.3: 创建工具脚本 ✅ 2026-04-29
- [x] Step 1.4: 创建最小测试 ✅ 2026-04-29

### 模块 2：配置系统
- [x] Step 2.1: 创建配置 schema ✅ 2026-04-29
- [x] Step 2.2: 实现 YAML loader 与 override ✅ 2026-04-29
- [x] Step 2.3: 创建默认配置文件 ✅ 2026-04-29
- [x] Step 2.4: 完善配置测试 ✅ 2026-04-29

### 模块 3：核心类型、support transform、player perspective
- [x] Step 3.1: 创建核心类型文件 ✅ 2026-04-29
- [x] Step 3.2: 实现空间 spec ✅ 2026-04-29
- [x] Step 3.3: 实现 support transform ✅ 2026-04-29
- [x] Step 3.4: 实现 PlayerPerspective 与 GameHistory ✅ 2026-04-29

### 模块 4：环境适配层
- [x] Step 4.1: 定义 GameAdapter 与 EnvFactory ✅ 2026-04-29
- [x] Step 4.2: 实现 GymnasiumAdapter ✅ 2026-04-29
- [x] Step 4.3: 实现 TicTacToe 与 Connect4 ✅ 2026-04-29
- [x] Step 4.4: 接入 EnvFactory ✅ 2026-04-29

### 模块 5：网络协议与基础模型
- [x] Step 5.1: 创建模型文件与输出类型 ✅ 2026-04-29
- [x] Step 5.2: 实现 BaseMuZeroNetwork ✅ 2026-04-29
- [x] Step 5.3: 实现 MLPNetwork ✅ 2026-04-29
- [x] Step 5.4: 实现 NetworkFactory 与空壳高级模型 ✅ 2026-04-29

### 模块 6：TreeStorage、PUCT、搜索策略
- [x] Step 6.1: 实现 TreeStorage ✅ 2026-04-29
- [x] Step 6.2: 实现 PUCT 公式 ✅ 2026-04-29
- [x] Step 6.3: 实现 SearchPolicy 与 PUCTPolicy ✅ 2026-04-29
- [x] Step 6.4: 实现 DiscreteActionSampler ✅ 2026-04-29

### 模块 7：标准 MCTS
- [x] Step 7.1: 实现 SearchRequest 与 MCTS 初始化 ✅ 2026-04-29
- [x] Step 7.2: 实现 root expansion ✅ 2026-04-29
- [x] Step 7.3: 实现 simulation loop ✅ 2026-04-29
- [x] Step 7.4: 接入 Dirichlet root noise 与 temperature ✅ 2026-04-29

### 模块 8：ReplayBuffer 与 TargetBuilder
- [x] Step 8.1: 实现 ReplayBuffer ✅ 2026-04-29
- [x] Step 8.2: 实现 TargetBuilder ✅ 2026-04-29
- [x] Step 8.3: 实现 PrioritizedReplayBuffer ✅ 2026-04-29
- [x] Step 8.4: 实现 ReanalyzeQueue 占位 ✅ 2026-04-29

### 模块 9：标准 MuZero Trainer
- [x] Step 9.1: 实现 loss 模块 ✅ 2026-04-29
- [x] Step 9.2: 实现 OptimizerFactory 与 WeightStore ✅ 2026-04-29
- [x] Step 9.3: 实现 Trainer.train_step ✅ 2026-04-29
- [x] Step 9.4: 实现基础 CheckpointManager ✅ 2026-04-29

### 模块 10：CLI 与基础 smoke tests
- [x] Step 10.1: 实现 CLI 主入口 ✅ 2026-04-29
- [ ] Step 10.2: 实现应用组装函数（SelfPlayWorker 就绪）
- [x] Step 10.3: 实现最小 self-play episode ✅ 2026-04-29
- [ ] Step 10.4: 实现 train/eval smoke tests

### 模块 11：InferenceBatcher 与 Batch MCTS v1
- [ ] Step 11.1: 实现 InferenceBatcher
- [ ] Step 11.2: 改造 MCTS 支持 batcher
- [ ] Step 11.3: 实现 BatchMCTS.run_batch
- [ ] Step 11.4: 接入 CLI benchmark

### 模块 12：EfficientZero heads、loss、配置开关
- [x] Step 12.1: 实现 EfficientZeroHeads ✅ 2026-04-29
- [ ] Step 12.2: 扩展 MLPNetwork 输出
- [ ] Step 12.3: 实现 EfficientZeroLoss
- [ ] Step 12.4: 接入 efficientzero config 与 smoke test

### 模块 13：Sampled MuZero 与连续动作支持
- [ ] Step 13.1: 实现 ContinuousActionSampler
- [ ] Step 13.2: 扩展 MLPNetwork 支持连续动作 dynamics
- [ ] Step 13.3: 修改 MCTS sampled mode
- [ ] Step 13.4: Pendulum sampled smoke test

### 模块 14：GumbelPolicy 低仿真搜索扩展
- [ ] Step 14.1: 实现 GumbelPolicy 基础
- [ ] Step 14.2: 接入 SearchPolicyFactory
- [ ] Step 14.3: 创建 Gumbel config
- [ ] Step 14.4: Gumbel integration test

### 模块 15：多人 value vector 完整验收
- [ ] Step 15.1: 强化多人 TargetBuilder
- [ ] Step 15.2: 强化 TreeStorage.backup 多人逻辑
- [ ] Step 15.3: 创建三人 toy env
- [ ] Step 15.4: 多人 integration test

### 模块 16：Checkpoint registry 与预训练权重接口
- [ ] Step 16.1: 实现 CheckpointMetadata
- [ ] Step 16.2: 实现 checkpoint inspect/export/import
- [ ] Step 16.3: 实现 checkpoint CLI
- [ ] Step 16.4: checkpoint roundtrip integration test

### 模块 17：Ray optional backend
- [ ] Step 17.1: 实现 ExecutorBackend 与 LocalExecutorBackend
- [ ] Step 17.2: 实现 EvaluatorWorker 与 ReanalyzeWorker 占位
- [ ] Step 17.3: 实现 RayExecutorBackend
- [ ] Step 17.4: 接入 backend factory 和 Ray smoke

### 模块 18：Benchmark、日志、最终工程验收
- [ ] Step 18.1: 实现 MetricsLogger 和 PerformanceTracker
- [ ] Step 18.2: 接入 Trainer 与 SelfPlay 日志
- [ ] Step 18.3: 完善 benchmark runner
- [ ] Step 18.4: 最终文档与全量验收

## 已知问题

（暂无）

## 进度更新规则

- Codex 每完成一个完整模块后，批量更新对应模块下所有 Step 的复选框。
- Codex 不需要每完成一个小 Step 就更新本文件。
- 如果遇到 `plan.md` 未覆盖的技术决策，先记录到 `docs/issues.md`，再停下报告。
