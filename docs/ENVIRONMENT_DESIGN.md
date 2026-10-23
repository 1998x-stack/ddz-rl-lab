# 斗地主多智能体环境设计规范（Environment Design）

> 状态：当前实现说明 + 下一阶段设计提案｜对应代码：src/ddz_core/、src/ddz_env/、src/ddz_web/arena.py  
> 已实现基线：ddz-env-v1 / ddz-observation-v1 / ddz-action-v1；项目版本 0.12.0。与 docs/GAME_RULES.md 一同构成环境语义的基础约定。

## 1. 目标、研究对象与不变量

DDZ-RL-Lab 使用三人、不完全信息、零和团队收益、轮流决策的无赌注斗地主研究环境。环境的职责是**保证对局正确且可重放**，而不是把错误动作交给 RL 学习。

**核心不变量：**

- **唯一合法动作源**：GameEngine.legal_actions()；DDZEnv.step(local_index) 必须验证当前局部索引合法。
- **信息隔离**：每个策略只收到 PlayerObservation，不得收到其他玩家完整手牌、未公开底牌、特权 GameState。
- **牌张守恒**：任何时刻 54 张物理牌恰好存在于手牌、已出牌，以及地主确定前的底牌之中。
- **决策版本固定**：规则、观察编码、动作编码、候选集合排序、模型 checkpoint 共同确定一次可重现决策。
- **区分终止与截断**：正常胜负 terminated=true；达到重新发牌保护上限 truncated=true，不计胜方；Arena 的 MAX_TURNS 是运行错误保护，不是自动判胜。
- **强拒绝语义**：任何非法动作不产生状态迁移，不能以默认 PASS 或自动“修正牌型”代替。
- **评估与训练一致**：状态、候选和角色路由使用相同代码路径，不能在 Eval 中使用另一个实现的牌序比较逻辑。

## 2. 分层结构和真实代码位置

~~~text
                         自博弈 Actor / Arena Orchestrator
                                      |
                       observe(acting_seat)
                                      v
                    ddz_core.state.PlayerObservation
                                      |
                        ddz_env.encoding.py
                         125D state + 51D candidate
                                      |
                  ddz_models.policy.RoleBundle.act()
                                      |
                           local legal index
                                      |
                              DDZEnv.step()
                                      |
                    ddz_core.engine.GameEngine.step()
                                      |
               rules.generate_plays / beats / physical cards
                                      |
                events / reward / replay / invariant check
~~~

| 层 | 现有文件 | 责任 |
|---|---|---|
| 规则 | ddz_core/cards.py, actions.py, rules.py | 牌编码、语义动作、牌型识别、比较、穷举生成 |
| 状态机 | ddz_core/state.py, engine.py | 隐藏信息、阶段转换、洗牌、完整性校验、回放摘要 |
| 环境 | ddz_env/environment.py | reset、observe、step、局部动作索引、reward |
| 兼容层 | ddz_env/pettingzoo_env.py, catalogue.py | AEC 与固定全局动作 ID 映射 |
| 特征 | ddz_env/encoding.py | 版本化观测与动作编码 |
| 训练调用 | ddz_training/actors.py | 使用当前玩家可见的观测和合法候选采样 |
| 比赛调用 | ddz_web/arena.py | 注册模型、比赛推进、事件和公共快照 |

**安全边界**：DDZEnv.engine 在 Python 内可被 orchestrator 访问，是受信任控制面，不应提供给 learner/policy；禁止将 GameState 直接序列化至玩家 API。

## 3. 状态机与迁移规则

~~~mermaid
stateDiagram-v2
    [*] --> CALL: 54张洗牌/发17+17+17/底3
    CALL --> CALL: DECLINE_CALL (未满三人)
    CALL --> CALL: 三人全不叫/重发牌
    CALL --> ROB: CALL
    ROB --> ROB: ROB或DECLINE_ROB/跳过不合格者
    ROB --> PLAY: claimant确认为地主/公开并合入底牌
    PLAY --> PLAY: PLAY或PASS/轮流行动
    PLAY --> PLAY: 连续两PASS/TRICK_RESET
    PLAY --> FINISHED: 任一玩家出完
    CALL --> FINISHED: 重发次数超限/TRUNCATED
    FINISHED --> [*]
~~~

当前阶段枚举 Phase.CALL、ROB、PLAY、FINISHED。实际动作 ActionKind 包含 CALL、DECLINE_CALL、ROB、DECLINE_ROB、PLAY、PASS。轮到谁行动通过 current_seat 判断；完成后 current_seat=None。

**每一条 transition 的事务次序**：

1. 在原状态上读取当前合法动作、当前座位、当前阶段。
2. 校验类型及 exact membership（DDZEnv.step 中校验局部索引，GameEngine.step 中再次校验 Action）。
3. 如不合法，立刻抛出 IllegalAction；不得增量写 step_index、events 或 hands。
4. 合法时更新座位、计数、手牌和公共事件，调用 assert_invariants()。
5. 生成包含 event、acting_seat、next_seat、rewards、terminated、truncated、state_digest 的 Transition。

事件序列编号从 1 开始；自动 DEALT / LANDLORD_GRANTED / TRICK_RESET / FINISHED 等事件可以由一次提交的 action 引发。**step() 返回本次最后一条事件**，若需要全部事件，必须按序列号读取新增范围，不能假设一动作恰好一个事件。

## 4. 物理状态与信息分层

### 4.1 仅引擎有权看到的 GameState

GameState 含 seed、完整 hands、bottom_cards、call/rob 内部状态、所有 played_by、last_play、passes、winner、step_index 与公共事件。其内部可用于回放一致性、离线调试、裁判校验；**不得作为训练样本特征、面向普通玩家的 JSON 或在线推理输入**。

### 4.2 每玩家 PlayerObservation

| 字段 | 来源 | 可见性/使用约束 |
|---|---|---|
| seat, phase, current_seat | 公共游戏状态 | 所有人可知 |
| own_physical_cards, own_counts | 自身手牌 | 仅该 seat 的策略/授权玩家 |
| remaining_counts | 3 位各自剩余张数 | 公共 |
| played_counts_by_seat | 各玩家已经打出的牌 | 公共 |
| public_bottom_counts | 地主确定后公开底牌 | 之前必须是 15 个零 |
| landlord, caller, claimant, first_bidder | 叫抢地主公共状态 | 公共 |
| last_play, last_play_counts, last_play_seat | 当前牌轮待压牌 | 公共 |
| public_events | 已脱敏的公开历史 | 事件顺序不可任意篡改 |
| legal_actions | 当前合法语义动作 | 仅轮到该 seat 时非空 |
| deal_index, step_index, winner | 进度、结果 | 不泄露私有发牌 |

**观测脱敏测试**：固定公开历史，改变未公开的其他玩家手牌，玩家观察中除其自己的手牌（应保持固定）以外不得出现对手物理牌 ID 或手牌计数细节；同一玩家能看到剩余张数属于允许的公共信息。

### 4.3 三种视图应隔离

- **PlayerObservation（现有）**：可信策略输入，可包含自己的牌。
- **PublicSpectatorView（Arena 的 public_snapshot 已有类似投影）**：公共事件、剩余张数、地主、桌面牌，不应展示未公开手牌或局内种子。
- **Audit/Replay（已有）**：用于受信任的离线审计，包含 seed、rules、完整动作列表与最终 digest；不得直接开放为实时玩家信息源。

如未来实现“全明牌 Debug Arena”，必须将其作为**明确单独的权限模式**与普通对局隔离，严禁把调试透明牌视图用于强化学习观测或比赛评测。

## 5. 动作空间设计：局部 index 与全局 ID

### 5.1 权威局部语义候选集

本地环境没有一个固定大小的 PLAY 输出空间。它为当前手牌及桌面牌生成完整候选：

~~~python
obs = env.current_observation()
candidates = obs.legal_actions  # tuple[Action, ...]
local_index = policy.act(obs, rng, epsilon=0).local_index
assert 0 <= local_index < len(candidates)
transition = env.step(local_index)
~~~

该 local_index **仅对生成它的这次观测有效**。经过任何动作、发牌或规则版本更新后，旧 index 不能复用。当前排序是确定性的，但绝不能把排序位置作为长期模型动作 ID。

### 5.2 PettingZoo 全局目录与动作掩码

现有 PettingZooDDZEnv 采用 **AEC**（不是 ParallelEnv），包装 fixed Discrete 全局目录：前 5 个语义动作为 CALL、DECLINE_CALL、ROB、DECLINE_ROB、PASS，后续是标准整副牌可生成的所有 PLAY。现有 V3 文档给出的目录大小为 **56,251**（任何版本变化应由 action_catalogue() 实际结果校验）。

- observation_space = Dict(observation: Box[125], action_mask: MultiBinary[全局目录长度])。
- infos[agent]["legal_action_ids"] 是当前有效全局动作编号序列。
- adapter.step(global_id) 先校验是否存在于当前选中 agent 的 legal_action_ids，再映射到 local_index。
- 非当前行动玩家的 mask 全零，不能据此断言该玩家没有牌可出。
- 训练/推理应优先使用**稀疏局部候选矩阵**，避免每次密集评分 56k 维全局动作。

**禁止混用**：将 local_index 直接传给 PettingZoo.step(global_id)，或将 global_id 直接传给 DDZEnv.step(local_index)，是严重的可复现性错误。

## 6. 特征编码与维度

当前 ddz_env.encoding.py：

| 编码 | 当前维数 | 字段 |
|---|---:|---|
| encode_observation | **125** | own hand 15、played 45、bottom 15、last play 15、remaining 3、seat one-hot 3、acting/landlord/caller/claimant 各4、first bidder 3、last play seat 4、phase 4、pass flag 1、redeal 1 |
| encode_action | **51** | kind one-hot 6、点数 15、pattern kind one-hot 14、主值 one-hot 15、chain 1 |

计数特征通常除以 4；剩余张数除以 20；所有编码值应该在 [0,1]。role_by_seat 的上下家关系在 policy.py 中按 (landlord + 1)%3=farmer_down、(landlord + 2)%3=farmer_up。

已知边界：125D 聚合向量没有独立的完整时间序列历史编码。虽然 PlayerObservation.public_events 保留公共历史，但当前 CandidateQNetwork **没有直接建模历史事件**。未来加入 HistoryEncoder 时必须更换 observation_version 与模型 checkpoint 契约，不可静默改变同版本特征顺序。

## 7. 奖励设计与多智能体博弈

当前仅使用终局奖励，中间每次 step 返回三方全零向量。正常结束时：

| 结果 | 地主 seat | 两名农民 seat（每人） | 总和 |
|---|---:|---:|---:|
| 地主获胜 | +2 | -1 | 0 |
| 农民获胜 | -2 | +1 | 0 |
| 截断 / 未完局 | 0 | 0 | 0 |

角色赋值取决于最后地主是谁，而不是固定 seat；不能把“farmer_down”误当永远属于 seat 1。

本表仅表达**团队效用与训练标签**，不涉及积分押注、金币或真实交易。若未来引入过程奖励（出完牌速度、配合等），必须独立报告它对策略目标的偏置，并保持终局胜率作为主评价量，避免模型学会“刷奖励”而牺牲真实胜率。

## 8. 随机性、重放和确定性

- GameEngine._new_state 通过 SHA-256 派生的 RNG 随 seed + deal_index 确定性洗牌并选择首叫位。
- export_replay() 使用 schema_version=replay.v1，保存 seed、RuleConfig、动作 JSON、state_digest、terminated/truncated、winner。
- replay(record) 从种子和规则重新启动，引擎依次 step 每个动作，并核对最终 SHA-256 digest。
- 相同种子**不等于**相同对局：必须相同 rules、动作序列和兼容的引擎实现。
- 强制明确随机数的来源：环境洗牌、epsilon 探索、历史对手采样、网络初始化、评测种子分别记录，避免跨进程 RNG 共享。
- 跑评测时使用独立且未参与训练/调参选择的 holdout seeds。

### 8.1 回放与在线 Arena 的边界

现有 Arena public_snapshot 在未结束时不公开 seed；已结束游戏可从 replay endpoint 导出回放。回放包含有能力恢复完整手牌的 seed，应按审计级敏感数据管理；未结束前不允许把 replay 包发给未授权客户端。

## 9. 环境调用示例（现有 API）

~~~python
from ddz_env.environment import DDZEnv

env = DDZEnv(seed=2026)
obs = env.reset(seed=2026)

while not env.done:
    obs = env.current_observation()
    # 纯演示：始终选择当前合法候选列表中的第一个动作
    transition = env.step(0)

print("last transition:", transition.terminated, transition.truncated)
print("team rewards:", env.rewards())
print("audit replay:", env.export_replay()["schema_version"])
~~~

示例的“第一个动作”并非优秀策略，也不是随机合法策略；它仅验证接口与合法动作约束。完整环境测试应调用 tests/test_env.py、tests/test_pettingzoo_adapter.py 及 GameEngine 相关测试。

## 10. 下一阶段扩展建议（**尚未实现**）

### 10.1 统一语义决策契约

提供内部 DecisionContext（request_id、game_id、expected_state_digest、seat、rule_id、observation_version、action_version、候选动作列表 hash）与 Decision（local_index、完整 Action、model_id、model_sha、耗时）。动作提交时由同一局锁重查状态摘要与 Action exact membership。索引仅做便利字段，不能代替完整语义动作。

### 10.2 可选 HistoryEncoder

可由 public_events 建立时序编码：事件类型、actor、PLAY counts、牌型、双方剩余数、轮次边界。采用固定窗口 + padding mask 或分层 transformer；确保只引入对玩家当时已公开的事件。训练与推理必须共享同一 HistoryEncoder 实现。

### 10.3 高吞吐环境与并行采样

- CPU worker 负责确定性牌局和候选枚举；GPU learner 负责批量候选评分。
- 进程间只传脱敏特征、候选、标签和非敏感度量，不共享完整 GameState。
- 通过可控的 seed 分区避免 worker 之间碰撞；要求中断恢复后单局 replay 一致。
- 缓存候选枚举时 key 至少含 hand_counts 与 RuleConfig，不能只按牌型缓存。
- 建议采集 legal candidate 数量直方图、枚举耗时、单局决策数、无效尝试计数，识别性能瓶颈。

### 10.4 Gym/PettingZoo 兼容验证

使用 PettingZoo 官方 API 测试或自建同等严格的 dead-step、agent_selection、reward 累积、action_mask 语义验证。引入新的多阶段 actor 时，不可擅自将 CALL/ROB 融入 PLAY action space。

## 11. 验收标准

| 领域 | 必测条件 | 结果要求 |
|---|---|---|
| 规则一致性 | 上家 22、下家 QQ | QQ 不出现在当前合法候选中；提交必须被拒绝 |
| 非法状态安全 | step(-1)、越界、bool、过时 index | 非法输入被拒绝，state_digest 不变 |
| 信息隔离 | 叫地主前读取底牌；读取其他 seat | 不可获取隐藏牌；非行动方 legal_actions=空 |
| 牌张守恒 | 多 seed 全阶段压力回放 | 54 张恰好守恒 |
| 回放 | 固定 seed、rules、完整 actions | 完全匹配 state_digest |
| 奖励 | 地主/农民获胜；重发截断 | ±2 / ±1 零和；截断全 0 |
| 空间 | 状态编码、动作编码、mask | 125D / 51D / 与目录严格对齐 |
| 三方顺序 | 叫抢阶段、连续两个 PASS、玩家出完 | seat 和阶段转换符合规则 |
| 训练集 | 任意 sample 的动作 | 来源于当时的 legal_actions |
| 漂移 | 改动 rules、encoder、pattern_key | 旧 bundle 不得未经校验直接使用 |

~~~bash
python -m pip install -e '.[dev,rl,train]'
pytest -q tests/test_env.py tests/test_pettingzoo_adapter.py tests/test_engine.py
pytest -q
~~~

## 12. 参考文档

- docs/GAME_RULES.md、docs/RULE_SPEC.md、docs/DATA_CONTRACTS.md
- docs/P2_ENV_DESIGN.md、docs/V3_ENV_CONTRACT.md
- docs/MODEL_TRAINING.md、docs/MODEL_INFERENCE_SERVICE.md
