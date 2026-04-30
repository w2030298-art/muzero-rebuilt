# Batch MCTS — 实现参考

## 来源
- 工程方案：CPU tree + GPU batched inference queue；参考 MCTX 的 batched search 思路，但不切换到 JAX。
- 核心贡献：减少 MCTS 中大量 tiny GPU inference 调用，提高 RTX 4060 Laptop 上的吞吐稳定性。

## 关键实现要点
- Tree selection、expansion、backup 第一版保留在 CPU。
- 神经网络推理统一进入 `InferenceBatcher`。
- `InferenceBatcher` 支持 initial 与 recurrent 两类请求队列。
- 队列达到 `batch_size` 后自动 flush。
- flush 时使用 `torch.no_grad()`；CUDA AMP 按 `precision` 配置启用。
- callback 对每个请求返回单样本 `NetworkOutput`。

## 对应模块
- 使用方：plan.md 中的模块 11、18
- 集成方式：先实现 InferenceBatcher，再把 MCTS 改造成可选使用 batcher，最后实现 BatchMCTS.run_batch 和 benchmark。

## 注意事项
- 第一版 `BatchMCTS.run_batch()` 可以顺序执行 MCTS，但必须共享同一个 batcher。
- 不要在第一版实现全 GPU tensorized tree。
- MCTS 不允许绕过 batcher 直接做大量单样本 GPU 推理。
