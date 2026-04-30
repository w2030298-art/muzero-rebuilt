# 多人游戏与 Value Vector — 实现参考

## 来源
- 架构设计决策：从底层支持 `num_players >= 1`，避免后期大范围重构。
- 核心贡献：单人、二人零和、多人 general-sum 使用统一 value/reward 结构。

## 关键实现要点
- `NetworkOutput.value` 支持 `[B]` 或 `[B, num_players]`。
- `TreeStorage.value_sum` 固定为 `[max_nodes, num_players]`。
- `GameHistory` 保存 `players` 和 reward vector。
- `PlayerPerspective.project_value()` 负责将 vector value 投影为当前玩家标量视角。
- 二人零和：`value[to_play] - value[1 - to_play]`。
- 三人以上：`value[to_play] - mean(value[others])`。
- `TargetBuilder` 对 vector reward/root_value 逐元素折扣累加。

## 对应模块
- 使用方：plan.md 中的模块 3、4、5、6、8、15
- 集成方式：先在核心类型中支持 value vector，再通过 TicTacToe、Connect4、three_player_toy 做验收。

## 注意事项
- 不要把二人零和标量作为底层唯一表示。
- 搜索 backup 可以使用当前玩家视角标量，但 replay/network target 必须保留 vector 能力。
- 三人 toy env 用于结构测试，不追求博弈复杂度。
