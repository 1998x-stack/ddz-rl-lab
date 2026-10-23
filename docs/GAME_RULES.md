# 斗地主出牌规则与合法动作规范（Game Rules）

> 状态：规范性文档（现行代码对齐）｜项目：DDZ-RL-Lab 0.12.0｜基准规则：classic_call_rob_2016_reference_v1  
> 真值源：src/ddz_core/cards.py、actions.py、rules.py、engine.py、state.py  
> 本项目是无赌注的离线强化学习研究模拟器，不包含下注、结算、金币、货币倍率。本规则是**项目冻结的规则变体**，不能等同于所有商业斗地主平台的当前玩法。

## 1. 必须成立的原则

1. **所有可执行出牌必须来自 GameEngine.legal_actions()**。UI、模型、训练器、回放器均不能自主宣布某张牌“合法”。
2. **先识别完整牌型，再比较大小**。不得按最大单张牌比较整组牌，也不能把附件四张牌误判为独立炸弹。
3. 相同普通牌型只有**种类、总张数、连续段长度匹配且主值严格更大**才能压过；任何普通牌型不能压炸弹。
4. 炸弹压普通牌与较小炸弹；王炸压所有牌型。王炸不可被压。
5. 不满足集合成员校验的动作必须原子拒绝，游戏状态、历史、计数均不改变。
6. 牌型歧义使用不同的 (pattern_kind, main_rank, chain_length) **语义动作**保留，不仅根据牌张集合去重。
7. 任何规则开关改变必须评估 RuleSpec、模型编码、动作目录、轨迹和回放兼容性，不能悄悄沿用旧数据。

## 2. 牌、座位和基础对局

- 物理牌总计 54 张：普通牌 52 张（13 点数 × 4 花色），小王 SJ 与大王 BJ 各 1 张。
- 点数顺序从小到大：**3 < 4 < 5 < 6 < 7 < 8 < 9 < 10 < J < Q < K < A < 2 < SJ < BJ**。
- 点数索引 0..14 对应上述顺序；物理 ID 0..51 按 card_id % 13 取点数，52=SJ、53=BJ。
- 三个 seat 依次为 0→1→2→0。每人初始 17 张；保留 3 张底牌，地主确定后公开底牌并放入地主手牌（地主 20 张）。
- 先叫地主/抢地主，再由地主首出。正常对局中任一玩家手牌清零即结束：地主先出完，地主队赢；任一农民先出完，农民队赢。
- 花色对本规则的牌力、牌型与决策无影响。语义出牌仅表示 15 维点数计数；规则引擎选择当前玩家手中确定性的物理牌 ID。

## 3. 阶段与叫抢地主规则

| 阶段 | 合法操作 | 阶段规则 |
|---|---|---|
| CALL | CALL / DECLINE_CALL | 随种子确定首叫位，按座位顺序轮流；第一个 CALL 结束叫牌；三人全部 DECLINE_CALL 则按确定性新发牌继续 |
| ROB | ROB / DECLINE_ROB | 第一个叫地主者先获得 claimant 身份；后续机会顺序：之后两名玩家、原叫地主者；此前 DECLINE_CALL 的座位和当前 claimant 跳过 |
| PLAY | PLAY / PASS | 地主首出；领出时不可 PASS；跟牌可以 PASS 或打更大的合法牌 |
| FINISHED | 无 | 已确定胜方或因最大重发次数达到上限而截断 |

ROB 会更新 claimant 与 rob_count，但不会计算资金、分数押注、货币倍率。所有自动转阶段均产生公共事件。重发次数超过 RuleConfig.max_redeals 时终局标记 truncated=true、winner=null，不计任何一方获胜。

## 4. 牌型清单（现行实现）

下表的“示例”均为点数表述，连续段仅允许 **3 至 A**，不能包含 2 或王。长度下限是项目实现定义，不可与其他变体混用。

| PatternKind | 中文名 | 张数/构成 | 主值及比较条件 | 示例 |
|---|---|---|---|---|
| SINGLE | 单张 | 1 | 自身点数 | Q |
| PAIR | 对子 | 2 张相同，普通牌 | 对子点数 | 99 |
| TRIPLE | 三张 | 3 张相同，普通牌 | 三张点数 | 777 |
| TRIPLE_SINGLE | 三带一 | 三张 + 不同主点数的一张 | 三张点数 | 777+Q |
| TRIPLE_PAIR | 三带二 | 三张 + 不同主点数的一对 | 三张点数 | 777+QQ |
| STRAIGHT | 顺子 | ≥5 个连续点数，每点数 1 张 | 最大连续点数；相同长度 | 78910J |
| PAIR_CHAIN | 连对 | ≥3 个连续点数，每点数 2 张 | 最大连续点数；相同连续段长度 | 778899 |
| TRIPLE_CHAIN | 飞机不带翼 | ≥2 个连续点数，每点数 3 张 | 最大连续点数；相同连续段长度 | 777888 |
| AIRPLANE_SINGLE | 飞机带单翼 | 连续 n≥2 个三张 + n 张符合规则的单牌 | 主三连的最大点数；相同 n | 777888+3+Q |
| AIRPLANE_PAIR | 飞机带对翼 | 连续 n≥2 个三张 + n 对符合规则的对子 | 主三连的最大点数；相同 n | 777888+33+QQ |
| FOUR_TWO_SINGLE | 四带二单 | 四张同点数 + 2 单牌 | 四张主值；同牌型 | 5555+8+9 |
| FOUR_TWO_PAIR | 四带两对 | 四张同点数 + 2 对不同点数 | 四张主值；同牌型 | 5555+88+99 |
| BOMB | 炸弹 | 恰好 4 张同点数（无附件） | 炸弹主值 | 9999 |
| ROCKET | 王炸 | SJ + BJ | 最大牌型，无牌能压 | SJ+BJ |

**特别注意：**
- 2 可以形成对子、三张、炸弹及部分附件，但**不能进入顺子、连对或飞机的主连续段**；两个王也不能组成对子。
- 飞机单翼不允许从飞机主三张的同点数取剩下的第四张作为翼；飞机对翼的对子点数必须互异，不能使用王。
- 四带两对的两对点数互异；四带二单按现行配置允许附件重复，例如 **5555+88** 被视作 FOUR_TWO_SINGLE，而不是 PAIR 或 BOMB。
- 四带二单、四带两对的四张部分不是可独立使用的炸弹；整组牌型仍为 FOUR_TWO_SINGLE / FOUR_TWO_PAIR。
- 默认禁止将大小王**同时**作为一组单翼附件的一部分；并非禁止单独以王作为合法单张。

### 4.1 规则配置（与代码一致）

| RuleConfig 字段 | 当前默认值 | 含义 |
|---|---|---|
| rule_id | classic_call_rob_2016_reference_v1 | 人可读规则标识 |
| repeated_airplane_single_wings | true | 飞机单翼允许同点数重复，但不能超过剩余张数 |
| repeated_four_two_singles | true | 四带二单允许两张附件为同点数 |
| forbid_double_joker_as_wings | true | 一组单翼附件不能同时包含 SJ 与 BJ |
| max_redeals | 100 | 最多重发次数保护；达到上限截断而非判负 |

以上配置属于 **house rule**；真实比赛或线上平台具体限制可能不同。若更改开关，必须给新版本独立测试样本、轨迹标识和模型兼容校验。

## 5. 比大小算法（权威次序）

输入为已经识别的完整 Pattern(candidate) 与当前待压牌的 Pattern(previous)。当前代码的判定相当于：

~~~python
def beats(candidate, previous):
    if previous.kind == ROCKET:
        return False
    if candidate.kind == ROCKET:
        return True
    if candidate.kind == BOMB:
        return previous.kind != BOMB or candidate.main_rank > previous.main_rank
    if previous.kind == BOMB:
        return False
    return (
        candidate.kind == previous.kind
        and candidate.total_cards == previous.total_cards
        and candidate.chain_length == previous.chain_length
        and candidate.main_rank > previous.main_rank
    )
~~~

比较时 **主值严格大于（>），不是大于等于**。不能靠点数和、最大附件、对子中的某一张、或模型 Q 值改变合法性。无最后待压牌时，允许任一满足手牌包含关系的合法 PLAY；有待压牌时，只允许能 beats 的 PLAY 加 PASS。

### 5.1 必须固定的判例

| 上家已出 | 本家尝试 | 判定 | 原因 |
|---|---|---|---|
| 22 | QQ | **非法** | Q < 2；即使同为对子也不能压牌 |
| QQ | 22 | 合法 | 同牌型且 2 > Q |
| 22 | SJ+BJ | 合法 | 王炸通杀 |
| 22 | 3333 | 合法 | 炸弹压普通对子 |
| 22 | 2 | **非法** | 单张与对子牌型不同 |
| 777 | 888 | 合法 | 同为三张、主值更大 |
| 777+Q | 888+3 | 合法 | 都是三带一；比较三张的主值 |
| 777+Q | 888 | **非法** | 牌型/张数不同 |
| 34567 | 45678 | 合法 | 顺子长度同为 5 且最大点数更大 |
| 34567 | 345678 | **非法** | 长度不同 |
| 3333 | 4444 | 合法 | 炸弹之间比主值 |
| 4444 | 3333 | **非法** | 较小炸弹不能压较大炸弹 |
| SJ+BJ | 任意牌 | **非法** | 王炸不可被压 |
| 首出（无 last_play） | PASS | **非法** | 领出不能 PASS |
| 跟牌（存在 last_play） | PASS | 合法 | 玩家主动放弃本次压制机会 |

这里的“非法”是相对于表格指定的上一手牌，未必意味着“手中的牌本身不是一个合法牌型”。

## 6. 一轮出牌（trick）和 PASS

- last_play、last_play_counts、last_play_seat 记录当前必须压制的最近有效出牌。
- 有效 PLAY 会更新 last_play 与归零连续 PASS 计数，回合轮到下一个 seat。
- 跟牌者可以 PASS；连续第一个 PASS 之后轮到下一位。
- 若在**同一有效出牌后**连续 PASS 两次，则清空 last_play，触发 TRICK_RESET，轮到上一有效出牌者自由领出。
- 游戏结束后 legal_actions 为空，禁止任何额外动作。
- 执行动作前，Rule Engine 校验 **完整 Action 对象是否属于当下的 legal_actions 集合**，包括点数计数和 pattern_key。

## 7. Action JSON 与歧义处理

完整语义 Action 格式（来自 src/ddz_core/actions.py）：

~~~json
{
  "kind": "PLAY",
  "counts": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0],
  "pattern_key": ["PAIR", 12, 1]
}
~~~

**上例仅示意字段格式，不能作为有效 PAIR 动作样例**：counts 中包含 Q 一张和 2 两张，规则引擎必须将其拒绝。正确的 **22** 应为：

~~~json
{
  "kind": "PLAY",
  "counts": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
  "pattern_key": ["PAIR", 12, 1]
}
~~~

JSON 数组固定 15 维，kind ∈ CALL、DECLINE_CALL、ROB、DECLINE_ROB、PLAY、PASS；非 PLAY 操作必须有全零 counts 且 pattern_key=null。rank、chain_length 是**零基主点数索引与链长度**，不代表物理卡 ID。

同一 15 维 counts 可能存在多个有效牌型解释（比如连续四个三张，也可能在允许重复单翼的变体下存在另一种飞机解释）。候选集合以 **(counts, pattern_key)** 而非 counts 唯一化，训练和推理必须保留所有解释；仅靠 15 维 counts 推断 pattern 的实现是不完整的。

## 8. 单一权威验证链路

~~~text
物理手牌
  -> cards_to_counts（15维）
  -> generate_plays(hand_counts, last_play, RuleConfig)
  -> 生成所有已识别且能压制的 PLAY
  -> 如为跟牌：加入 PASS
  -> GameEngine.legal_actions()（唯一执行白名单）
  -> 训练策略/推理策略只选白名单内的 action 或 local_index
  -> GameEngine.step(action) 再次验证完整集合成员资格
  -> 扣牌 / 更新 last_play / 生成事件 / 检查不变量
~~~

- 禁止前端、训练脚本和推理服务各自写一套“近似合法”比较器作为执行依据。
- 禁止先由模型输出任意 15 维计数，再仅尝试用惩罚项推动模型学会遵守规则。
- 对非确定性的客户端输入，只能反序列化为候选并做 exact membership；不能“智能补牌”、静默重解释牌型或错误降级成 PASS。
- 物理牌保守约束：所有手牌 + 已打牌 + （未发地主前的）底牌，合计必须恰好是 54 张无重复卡牌。

## 9. 验收矩阵与回归策略

**已有回归入口：** tests/test_cards.py、tests/test_patterns.py、tests/test_engine.py、tests/test_fuzz.py；详细逻辑来自现行源码，本文没有运行远端 CI。

建议增加以下测试层次，标为**待实施建议**：

1. 按 14 类牌型 × 合法/非法构造判例，覆盖最短/最长连续段、A/2/王边界、附件同点数开关。
2. 针对每种普通牌型，测试同种等长、同种异长、不同种、主值等于/低于/高于三维对比；单独覆盖炸弹和王炸。
3. **强制黄金用例：22 对 QQ 必须拒绝；QQ 对 22 必须接受**，在规则层、环境 step 层、服务推理层各测一次。
4. 对任意状态，所有 generate_plays 结果均是当前手牌子多重集；所有带 last_play 结果都必须满足 beats。
5. fuzz/property-based：非法动作执行前后 state_digest 相同，牌张守恒，每次真实 action 只使已打出的牌数正确增加。
6. 按 seed+rules+完整动作列表重放，得到相同 state_digest；在断言前对任意被篡改动作拒绝。
7. 构造歧义解释数据，确认模型、目录映射与 replay 不丢失 pattern_key。
8. 规则更改时将对照数据与预期结果一同升级为新的 ruleset fixture，不直接覆盖旧版基准。

运行现有测试：

~~~bash
python -m pip install -e '.[dev]'
pytest -q tests/test_cards.py tests/test_patterns.py tests/test_engine.py tests/test_fuzz.py
pytest -q
~~~

## 10. 版本边界与关联资料

- **当前已实现**：RuleConfig、完整候选枚举、beats、原子非法动作拒绝、确定性洗牌与 replay.v1。
- **需要持续验证**：代码注释称候选“穷举”，但穷举正确性不能仅靠生成器自身断言，应增加独立牌型 oracle、边界回归与 fuzz 证据。
- **规划中的改进**：规范化 rule-config SHA-256、显式规则变更迁移、完整的跨服务语义动作协议和更系统的对照平台规则核验。
- 关联：docs/RULE_SPEC.md、docs/DATA_CONTRACTS.md、docs/ENVIRONMENT_DESIGN.md、docs/MODEL_TRAINING.md、docs/MODEL_INFERENCE_SERVICE.md。
