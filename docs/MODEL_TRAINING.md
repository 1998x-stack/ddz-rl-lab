# 斗地主强化学习模型与训练体系（Model Training）

> 状态：**当前可运行基线 + 明确标注的后续研究计划**｜对应代码：src/ddz_models/、src/ddz_training/、src/ddz_eval/、src/ddz_quality/  
> 当前算法名称：Role-specific Deep Monte Carlo (DMC) Q(s,a) regression；**不是 PPO、DQN TD 更新，也不是 DouZero 的完整等价实现**。项目为离线无赌注模拟器。

## 1. 训练目标与基本假设

研究目标是提升三人斗地主环境中的**地主胜率、农民配合胜率与整体双侧表现**，而非仅让训练 MSE 下降。环境是三人不完全信息团队博弈；候选动作集合随手牌、牌型和上手出牌动态变化。

不可修改的训练约束：

1. **每一条训练 Action 都取自当时 GameEngine.legal_actions()**，从来不把非法动作当作可执行负样本。
2. 只有 PlayerObservation 可进入网络，绝不将 GameState.hands、未公开底牌等信息送到模型。
3. 同一轮策略网络看到的候选集与环境实际 step 的候选集一致；训练和推理共享 encode_observation / encode_action。
4. 对局截断不会产生虚构胜负标签；现有 actor 对截断局不产生 Experience。
5. 训练评估使用角色分层和独立样本统计，不能用单一自博弈胜率或训练损失代替强度证据。
6. Checkpoint 必须冻结 observation/action encoder、network schema、rule config 等兼容元数据；变更规则不能沿用不兼容轨迹。

## 2. 已实现的数据流

~~~text
Seed plan
   |
   v
多进程 CPU actors (spawn)
   |-- DDZEnv -> GameEngine -> legal_actions
   |-- PlayerObservation (脱敏)
   |-- RoleBundle.act(obs, epsilon) 仅从合法候选中挑选
   |-- env.step(local_index) 再次验证
   |-- 记录 [role, encoded_state, encoded_action, actor_seat]
   |
   v
完整非截断对局 -> 按实际赢家回填每条动作的终局回报 G
   |
   v
ReplayBuffer(capacity=50000 默认)
   |
   v
按 role 分组 -> MSE(Q_role(s,a), G) -> Adam -> clip_grad_norm_(5)
   |
   v
冻结候选网络 -> 评估 -> best/latest/history checkpoints
   |
   v
校验/登记模型 -> 离线锦标赛 -> 受控晋升 -> Arena 使用
~~~

**当前 Experience 字段**：role（字符串）、observation（125D tuple）、action（51D tuple）、actor_seat（0/1/2）、return_value（float）。Actor 不导出隐藏手牌；ReplayBuffer 仅存脱敏编码后的状态、动作及最终团队收益。

## 3. 网络结构与四角色设计

### 3.1 CandidateQNetwork（当前版本）

输入不是“对整个 56k 动作目录进行 softmax”，而是单个状态向量和可变数量 K 的当前合法候选矩阵：

- 状态向量：**125** 维，src/ddz_env/encoding.py 的 ddz-observation-v1。
- 单个动作向量：**51** 维，ddz-action-v1。
- State encoder：Linear(125,h) → ReLU → Linear(h,h) → ReLU。
- Action encoder：Linear(51,h) → ReLU。
- Score head：concat(state_emb, action_emb) → Linear(2h,h) → ReLU → Linear(h,1)。
- 默认隐藏维度 **h=128**；每个候选输出一个未校准的实数 Q 值，不是胜率概率。
- 推理时 state_encoder 只算一次，向 K 个合法动作复用同一个 state embedding。

公式：

\[
z_s=f_\theta(\mathrm{obs}), \quad z_a=g_\theta(\mathrm{action}),
\qquad Q_\theta(s,a)=h_\theta([z_s;z_a])
\]

只对当局被允许的候选集合 \(A_{\text{legal}}(s)\) 进行选择：

\[
a^*(s)=\operatorname*{argmax}_{a\in A_{\text{legal}}(s)}Q_\theta(s,a)
\]

网络分数不能推翻规则引擎的合法性；更高 Q 值并不意味着“允许 Q 对压 2 对”。

### 3.2 角色模型路由（当前版本）

| RoleBundle 键 | 决策阶段 | 路由规则 |
|---|---|---|
| bid | CALL / ROB | 当前处于叫抢地主阶段的任何 seat |
| landlord | PLAY | 玩家为最终地主 |
| farmer_down | PLAY | seat == (landlord + 1) % 3 |
| farmer_up | PLAY | seat == (landlord + 2) % 3 |

这四个角色各有一个独立 CandidateQNetwork 和优化器。**角色不固定绑定物理 seat**；不同对局地主随机变化。农民彼此协作，但目前是按局部可见状态分角色学习，不是显式通信/集中训练的联合策略。

### 3.3 当前策略探索

RoleBundle.act(obs, rng, epsilon) 先对全体合法候选计算有限 Q 值，再以概率 epsilon 在候选中均匀随机探索，否则取 argmax。epsilon 取值 [0,1]，比赛与 Arena 默认使用 0。非有限分数会抛出 FloatingPointError；动作集合为空时禁止执行。

注意：epsilon-greedy 随机动作同样只来自合法候选集，不会访问 15D 任意动作空间。

## 4. 当前训练目标（Monte Carlo 回归）

本仓库的训练使用**完整对局结束后回填的终局团队回报**，非 TD bootstrapping：

\[
G_i = \begin{cases}
+2,&\text{地主胜、当前 seat 是地主}\\
-2,&\text{农民胜、当前 seat 是地主}\\
+1,&\text{农民胜、当前 seat 是农民}\\
-1,&\text{地主胜、当前 seat 是农民}
\end{cases}
\]

每条 Experience 只包含 actor 实际采取的那一手及 G，网络回归损失是：

\[
\mathcal L_\theta = \frac{1}{B}\sum_{i=1}^{B}
 \big(Q_\theta(s_i,a_i)-G_i\big)^2
\]

每个 role 的 minibatch 独立更新；优化器是 Adam，默认学习率 3e-4，梯度范数上限 5。当前没有策略熵项、target network、GAE、PPO clip 或 DQN 的 Bellman 目标，不能在实验报告中错误标注。

**统计注意**：虽然体验缓存里一局可生成很多 Experience，但同一局动作共享最终胜负标签，因此各动作样本**不是统计独立的比赛结果**；胜率置信区间应按完整 deal/独立训练 run 聚类，而不是按动作数计算。

## 5. 默认配置与调参边界

当前 TrainConfig 重要参数：

| 参数 | 默认值 | 作用与主要风险 |
|---|---:|---|
| episodes | 100 | 每次 train 调用增加的对局数；不等于累计 epoch |
| batch_episodes | 8 | 累积多少完整对局后更新一次 |
| eval_games | 16 | 每个角色侧的评测 deal 数 |
| eval_every | 40 | 达到该已见局数倍数时评估 |
| seed | 2026 | 网络、采样与环境种子基准 |
| hidden | 128 | 每个角色网络隐藏宽度 |
| lr | 0.0003 | Adam 学习率 |
| epsilon | 0.15 | 当前 actor 的合法动作探索率 |
| max_steps | 1200 | 单局决策上限；超出抛错而非判负 |
| device | cpu | 当前默认 learner 设备 |
| actor_workers | 1 | CPU actor 进程数；>1 使用 spawn |
| replay_capacity | 50000 | Experience 缓存容量（决策数） |
| replay_sample_size | 2048 | 每次学习最多抽取样本数 |
| updates_per_batch | 1 | 每个更新批次中的 learner 更新次数 |
| history_capacity | 6 | 保留历史推理模型快照的数量基线 |
| history_probability | 0.35 | 选择历史模型作为一侧对手的概率 |
| history_keep_top | 0 | 历史模型最优保留数；0 表示不另加保护 |
| early_stop_patience_evals | 0 | 连续无提升评测容忍度；0 禁用 |
| min_score_improvement | 0.0 | best_score 需要超过的最小幅度 |

具体字段定义以 src/ddz_training/trainer.py 为准。默认 output 在 TrainConfig 中为 runs/dmc-v5；CLI 的缺省 output 为 runs/dmc-v4，因此生产实验应**显式传 --output** 避免混乱。训练 checkpoint resume 会拒绝核心参数发生变化（例如 lr、hidden、epsilon）。

## 6. 完整运行命令（现有 CLI）

安装：

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,rl,train]'
pytest -q tests/test_training.py
~~~

最小训练 smoke（并非真实强度证据）：

~~~bash
ddz-train train \
  --episodes 8 --batch-episodes 2 \
  --eval-games 4 --eval-every 4 \
  --seed 2026 --hidden 128 \
  --actor-workers 1 \
  --output runs/smoke-dmc
~~~

长运行基线（参数是实验建议，不宣称保证更强）：

~~~bash
ddz-train train \
  --episodes 10000 --batch-episodes 16 \
  --eval-games 256 --eval-every 1000 \
  --seed 2026 --hidden 128 --lr 0.0003 \
  --epsilon 0.15 --actor-workers 4 \
  --replay-capacity 50000 --replay-sample-size 2048 \
  --updates-per-batch 2 --history-capacity 6 \
  --history-probability 0.35 --history-keep-top 3 \
  --output runs/dmc-seed2026
~~~

从同配置 checkpoint 继续：

~~~bash
ddz-train train \
  --checkpoint runs/dmc-seed2026/latest.pt \
  --episodes 1000 --batch-episodes 16 \
  --eval-games 256 --eval-every 1000 \
  --seed 2026 --hidden 128 --lr 0.0003 \
  --epsilon 0.15 --actor-workers 4 \
  --replay-capacity 50000 --replay-sample-size 2048 \
  --updates-per-batch 2 --history-capacity 6 \
  --history-probability 0.35 --history-keep-top 3 \
  --output runs/dmc-seed2026
~~~

注意 resume 需要与 checkpoint 的不可变字段精确匹配，调节学习率不是 resume 兼容操作。对独立训练 seeds 使用独立输出目录；训练/评估日志不要混写。

评估单个冻结模型：

~~~bash
ddz-train evaluate \
  --checkpoint runs/dmc-seed2026/best.weights.pt \
  --eval-games 256 --seed 510000
~~~

**重要解释**：src/ddz_eval/tournament.py 的 evaluate_bundle() 在 CALL/ROB 使用固定规则下的 seeded random baseline，分别评估“模型地主对随机农民”和“模型双农民对随机地主”，并不是四角色模型的完整学习叫地主策略实测效果。

## 7. 数据采样、回放缓存与历史对手

### 7.1 Actor 采样

当前 actor 路径：读取冻结最新模型与可选历史版本，逐局生成 seed；每步生成 PlayerObservation 和候选集；探索或贪心选择合法 local_index；执行 env.step；结束后将结果归属到实际 actor_seat。若因重发上限 truncated=true，该局 samples 为空。

当随机选择历史对手后，**CALL/ROB 仍由当前模型统一训练**；地主确定后才按照 learn_side=landlord/farmers 切换一侧使用 frozen 对手。这样不会错误绑定未知地主阶段的固定 seat。

### 7.2 ReplayBuffer

- 容量是决策样本条数，不是 episode 数。
- 现有结构为 FIFO deque，采样时不放回均匀随机抽取最多 replay_sample_size 条。
- 没有优先级回放（PER）、样本重要性权重或显式类别均衡。
- 保存 checkpoint 时可以包含 replay 缓冲快照以便恢复；网络部署必须使用 inference_only 权重，不导出训练 replay。

### 7.3 数据清洗约束

下列检查建议加入训练采集/离线重放门禁（**尚未全部实现**）：

- 记录每条样本产生时的 rule hash、encoder version、policy hash、deal seed、step_index、actor role、合法候选集合 fingerprint（注意私有样本的访问限制）。
- 针对 sample 所标 Action 在原始回放对应状态重放，并检验 exact membership；发现一次不一致即隔离整局。
- 训练与验证 seeds 严格不重叠；所有历史对手 checkpoint hash 可追溯。
- 禁止对超时、崩溃、无限循环、max_redeals 截断局赋予“默认失败”标签。
- 同一物理手牌可能有不同合法 pattern_key；不可对相同 counts 强行唯一化。

## 8. 评估指标与选择协议

### 8.1 当前可用评估

src/ddz_eval/tournament.py 已实现：

\[
\mathrm{balanced\_score}
=\frac{\mathrm{landlord\_win\_rate}+\mathrm{farmer\_win\_rate}}{2}
\]

指标按双方比赛分别统计；同 seed 的两侧比赛具有相关性，不能把它们误当两笔独立 deal。还应同时报告 landlord_games、farmer_games、truncated、训练吞吐与多 run 方差。

src/ddz_eval/league.py 与 selection.py 支持对冻结合法随机基线/指定历史对手的成对对照，并以**完整 deal seed 为重采样聚类单位**估计相对于 incumbent 的差值 CI。

### 8.2 晋升门槛（现行评估逻辑）

- 不重用训练/调参集来给最终产品版本做显著性证明。
- Candidate 与 incumbent 在同样 deal seeds、对手、地主/农民侧进行 paired fixtures。
- 至少 min_deals（默认 30）个独立 deal clusters，且挑战者减 incumbent 的 bootstrap CI **下界严格大于 min_delta**（默认 0），才可以被推荐。
- 推荐不等于自动发布：ModelRegistry.promote 需由受信任 operator 执行，并校验模型 hash、现任 incumbent 和报告的证据。
- 多候选、多次盯同一 holdout 会造成选择偏差；应另设最终确认集、冻结比较计划。

**重要边界**：现有 league/selection 比赛同样使用固定随机叫牌；它不能证明 bid 子模型提升。若要评价 bid，需单独设计完整学习叫抢地主交叉对局，在各方同等可行配置下比较。

### 8.3 建议的指标面板

| 层面 | 主指标 | 辅指标 |
|---|---|---|
| 游戏正确性 | 非法已执行动作数 = 0 | 牌守恒失败、轨迹重放失败、坏牌型个数 |
| 对局强度 | landlord / farmers 双侧胜率 | 对手分层、配合交互、各角色均衡 |
| 训练稳定性 | 独立 seeds 的胜率分布 | MSE、梯度范数、非有限损失、训练曲线 |
| 吞吐 | decisions/s、episodes/s | Actor wall time、候选动作数、GPU 利用率 |
| 样本质量 | 完整非截断局比例 | role 样本占比、replay 新鲜度 |
| 可复现性 | 相同固定评测 seeds 结果一致 | 网络 hash、规则版本、数据契约版本 |

## 9. Checkpoint、完整性与模型资产

现有输出：

~~~text
runs/<experiment>/
  latest.pt                  # 带 optimizer、RNG、replay 的训练恢复包
  latest.pt.json             # SHA-256 sidecar
  best.weights.pt            # 纯推理 bundle（含 4 个 role）
  best.weights.pt.json       # 推理哈希与 encoder/network 元数据
  history/
    step-NNNNNNNNN.weights.pt
    step-NNNNNNNNN.weights.pt.json
  evaluations.jsonl
  throughput.jsonl
~~~

- load_checkpoint 校验 checkpoint SHA-256、schema、state_dim、action_dim、hidden、encoder versions、完整四角色参数集合。
- torch.load(..., weights_only=True) 限制反序列化风险，但**仍仅加载受信任来源的本地 checkpoint**。
- ModelRegistry 验证 inference_only=true、来源路径边界及不可变模型 ID，不能直接从网络用户上传任意 .pt 作为执行代码。
- 现行 checkpoint 元数据没有独立绑定 canonical RuleConfig 哈希和具体 Git 代码版本，这是**后续应补齐的兼容性缺口**；仅 encoder v1 相同不一定证明规则语义一致。

## 10. 研究阶段规划（**以下为建议，非仓库已实现**）

### Stage T1：先建立规则无错数据基线

- 引入规则 golden cases 与独立 oracle；强制阻断 22 被 QQ 压过的任何轨迹。
- 每条 Experience 增加可验证的 rule/encoder/checkpoint 指纹与原始 action 语义可追溯性。
- 离线重放全部训练数据中抽样的完整局，在状态不匹配时拒绝进入 learner。
- 把 legal-action violation 设为训练流水线的**发布阻断项**，不是平均性能指标。

### Stage T2：建立稳定的多种子 DMC 基线

- 固定训练配置与 seed 集，至少 3 个独立训练 run 做探索性比较，必要时增加独立 run。
- 将角色别 batch 分布、梯度范数、预测值范围与模型间方差画在一份报告中。
- 消融 replay/historical opponents/epsilon/hidden，对每个改变保留独立对照。
- 自博弈中加入合法启发式、冻结旧模型和随机对手以降低只会打某一策略的风险。

### Stage T3：丰富时序与协作表征

- 引入只基于公开事件的 HistoryEncoder（RNN/Transformer），对历史公开牌路形成可变长表示。
- 对双农民协作测量成对同局多策略替换的 interaction effect；使用 docs/V12_TRAINING_QUALITY.md 中的现有研究工具作为基础。
- 不允许通过自定义 history 拼接泄露对手实际手牌或未来事件；运行时特征与离线训练严格同构。

### Stage T4：探索更高阶学习算法

- 可考虑价值基准、Double Q/target network、n-step、优势加权行为克隆、PPO 或 actor-critic，但要**先明确各算法适合的 action scoring 机制**。
- PPO 若采用合法候选策略头，需要对每个状态的 K 个候选归一化概率，记录行为策略 logprob，并在重新采样时校验候选集指纹。不能直接对固定 56k 输出做未经约束的更新。
- 不完全信息自博弈可研究策略池/league，防止过拟合单一对手；真实泛化必须通过冻结对手和 holdout 验证。
- 任何新算法单独取模型/观察/动作版本及新实验名，不覆盖已有 DMC 的客观表现记录。

### Stage T5：训练到推理一体化

- 用相同的 action encoder、state encoder、role router 进行 train/serve golden parity 测试。
- 限定推理延迟与错误预算，批量评分但保持**每一局的合法候选隔离**，拒绝跨 session 复用局部 index。
- 模型候选经 registry、完整性校验与比赛准入后才能进入在线 Arena；服务出错时采取 fail-closed/暂停对局，绝不随意出牌。

## 11. 回归测试与验收

**现有测试**：

~~~bash
pytest -q tests/test_training.py tests/test_v5_pipeline.py \
  tests/test_v6_registry.py tests/test_v6_selection.py \
  tests/test_v11_reliability.py tests/test_v12_quality.py
pytest -q
~~~

建议补充**未实现的验收集**：

1. **训练动作来源一致性**：从 action logs 中取任何一条 Experience，对原局在同一 step 验证其 Action 属于 legal_actions。
2. **压牌黄金用例**：当上一手为 22 时，actor 绝不能执行 QQ；UI、Train、Arena 均复核。
3. **模型不变量**：K=1、K 很大、极端合法牌、所有角色均能 score_legal 且 argmax 在 [0,K)。
4. **编码版本契约**：125/51 维、数值有效、角色/seat 映射准确、规则 hash 一致。
5. **收敛与数据统计**：MSE 可下降但不能自动宣称胜率上升；独立对局评测无显著回退才晋升。
6. **断点续训**：同 seed、同参数、同候选模型版本可恢复；更改网络形状/optimizer 合约必须拒绝。
7. **信息泄露红线**：训练样本 JSON 不含其他 seat 手中物理牌或未来公开事件。
8. **统计有效性**：不以每动作样本数代替独立游戏数，不重复使用 holdout 证明提升。

## 12. 文档与实现追踪

- 规则与合法性：docs/GAME_RULES.md、src/ddz_core/rules.py
- 环境与观察：docs/ENVIRONMENT_DESIGN.md、src/ddz_env/encoding.py
- 原有训练版本：docs/V4_TRAINING.md、V5_TRAINING_AND_TOURNAMENT.md
- 晋升/回归：docs/V6_MODEL_SELECTION.md、V11_RECOVERY_AND_EVALUATION.md、V12_TRAINING_QUALITY.md
- 推理部署边界：docs/MODEL_INFERENCE_SERVICE.md
