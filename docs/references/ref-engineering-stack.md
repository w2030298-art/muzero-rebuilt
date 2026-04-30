# 工程栈与执行约束 — 实现参考

## 来源
- 阶段 1 技术调研与阶段 2 架构设计
- 核心贡献：固定工程工具链，避免 Codex 在实现中自行选型。

## 关键实现要点
- Python 固定为 `>=3.11,<3.12`。
- 项目使用 `uv` 管理依赖。
- CLI 使用 Typer。
- 配置使用 Pydantic v2 + PyYAML。
- 环境 API 使用 Gymnasium，必须保留 `terminated` 与 `truncated`。
- AMP 使用 `torch.amp.autocast("cuda")` 与 `torch.amp.GradScaler("cuda")`。
- Ray 只作为 optional extra，且只能在 `src/muzero/execution/ray_backend.py` 中 import。
- `torch.compile` 默认关闭，只能配置 opt-in。

## 对应模块
- 使用方：plan.md 中的所有模块
- 集成方式：模块 1 固定工具链，后续所有模块遵循该约束。

## 注意事项
- 不要引入未在 plan.md 指定的核心依赖。
- 不要将 Ray 作为默认依赖。
- 不要把旧 Gym API 的 `done` 当成唯一终止信号。
- 每个模块都必须通过 ruff、pyright、pytest 验证。
