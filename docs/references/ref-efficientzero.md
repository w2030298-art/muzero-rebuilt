# EfficientZero — 实现参考

## 来源
- 论文/项目：EfficientZero / EfficientZero V2
- 核心贡献：在 MuZero 基础上增强样本效率，引入 value prefix、consistency loss、reanalyze/off-policy 思路，并扩展到离散和连续控制。

## 关键实现要点
- `EfficientZeroHeads` 包含 `value_prefix()`、`projection()`、`prediction_projection()`。
- `EfficientZeroLoss` 与 `MuZeroLoss` 分离。
- value prefix loss 使用 MSE。
- consistency loss 使用 negative cosine similarity：`2 - 2 * cosine_similarity.mean()`。
- `use_value_prefix`、`use_consistency_loss`、`use_reanalyze` 必须由配置控制。
- 第一版只实现 value prefix 与 consistency loss；reanalyze 只保留接口和队列。

## 对应模块
- 使用方：plan.md 中的模块 12、17
- 集成方式：在 `MLPNetwork` 中按配置挂载 EfficientZero heads，在 Trainer 中按配置额外加权 loss。

## 注意事项
- 不要在第一版中强行实现完整 reanalyze。
- 不要把 EfficientZero loss 混进标准 MuZeroLoss。
- 每个新增 loss 必须有独立单元测试。
