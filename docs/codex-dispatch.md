# Codex 任务派发

进入连续执行模式。按以下流程工作：

## 启动
1. 读取 `docs/progress.md`，定位当前未完成的第一个步骤
2. 读取 `docs/plan.md`，加载完整计划
3. 理解项目现状：这是一个 RL-MEC benchmark 系统，含 17 个算法、GameTheory 环境、实验编排系统

## 执行规则
- 从当前进度点开始，按 plan.md 的模块顺序和步骤顺序逐步执行
- 每个步骤完成后运行该步骤的验证命令
- **验证通过 → 直接执行下一步骤，不要停下来问我**
- **验证失败 → 自行诊断修复，最多重试 2 次；仍失败则记录到 docs/issues.md 并停下报告**
- 每完成一个完整模块后，批量更新 `docs/progress.md`

## 模块间依赖提示
- 模块 1（VSCode）和模块 2（Registry 去重）互相独立，可先做任一
- 模块 3（God File 拆分）独立，不依赖模块 1/2
- 模块 4（Trainer 重构）独立
- 模块 5（适配器 Gym 合规）建议在模块 3 之后做
- 模块 6（进程管理）独立
- 模块 7（测试/文档扫尾）必须在模块 1-6 全部完成后做

## 关键技术要点

### 模块 1 — VSCode 面板
- 用 VSCode `inputs` 的 `pickString` 类型实现下拉选择，替代逐算法硬编码
- `env.PYTHONPATH` 必须保留在每个 launch 配置中（settings.json 的 extraPaths 只影响语言服务器）
- tasks.json 用 `type: "process"` 而非 `"shell"`（跨平台兼容）

### 模块 2 — Registry 去重
- 只迁移 RL 算法映射到 registry，Heuristic 算法（Greedy/Random/Local-only/Full-offload）保留在 benchmark.py 本地
- `import_agent` 方法用 `importlib.import_module` + `getattr` 动态加载

### 模块 3 — God File 拆分
- 每迁移一个类后立即运行 import 验证，再迁移下一个
- 注意 `game_theory_env.py` 开头的 `sys.path.insert` — 各子模块如果需要跨包 import，也要处理
- 循环依赖用参数传递解决（依赖注入），不要在子模块间互相 import
- 拆分后在 `__init__.py` 重导出所有类，保持向后兼容

### 模块 4 — Trainer 重构
- 只抽取数据处理逻辑，不改变训练流程
- `_detect_env_properties` 在 `BaseTrainer.__init__` 末尾调用，子类不再需要重复检测

### 模块 5 — 适配器合规
- 检查 `step()` 返回值是否为 Gymnasium 五元组 `(obs, reward, terminated, truncated, info)`
- 如果底层 `BaseMECEnv` 返回旧版四元组，在适配器中转换

### 模块 6 — 进程管理
- `process.wait(timeout=N)` 在 Windows/Linux 都可用
- grace period 改为 15 秒，可通过环境变量 `STOP_GRACE_PERIOD` 自定义

## 仅以下情况停下
- 验证失败重试 2 次仍无法解决
- 遇到 plan.md 未覆盖的技术决策
- 需要我提供外部资源（密钥、凭证、设计稿等）
- 前置依赖模块未完成

## 禁止行为
- 不要每完成一个小步骤就停下来请求确认
- 不要偏离 plan.md 自行添加功能
- 不要引入 plan.md 未指定的依赖
- 不要"优化" plan.md 已确定的方案
- 不要修改 RL 算法的训练逻辑或超参数

## 完成后
输出完成报告：
- 各模块完成状态表格
- 遇到的 issues 列表（如有）
- 需要用户后续处理的事项（如有）

现在开始执行。
