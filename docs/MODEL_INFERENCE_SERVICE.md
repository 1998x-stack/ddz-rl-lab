# 斗地主模型推理与部署服务规范（Model Inference Service）

> 状态：**现有 Arena 进程内推理说明 + 未来独立推理服务设计提案**  
> 对齐代码：src/ddz_models/policy.py、network.py；src/ddz_training/checkpoints.py；src/ddz_registry/registry.py；src/ddz_web/arena.py、api.py、cli.py。  
> 基准版本：ddz-candidate-mlp-v1 / ddz-observation-v1 / ddz-action-v1，项目 0.12.0。  
> 安全模式：无赌注、受信任环境策略服务；不可向未授权用户暴露其他玩家手牌、种子或推理数据。

## 1. 服务职责与硬性安全边界

推理服务的职责是：**输入当前行动者经过授权的观测与权威合法候选，使用已验证模型对候选评分，返回一条可执行的完整语义动作**。推理模型不是规则引擎。

**不允许发生的行为：**

- 把 15 维任意牌型向量、语言模型文本回答或前端点击结果直接当成可执行 PLAY。
- 从全部物理牌、固定 56k 全局动作目录或网络输出中跳过合法性掩码直接采样。
- 使用单牌最大点数推断对子/三带一等是否大于上一手，导致 **22 被 QQ 压过**。
- 缺少合法候选时擅自构造 PASS（领出时不能 PASS），或把推理异常无声转成随机出牌。
- 将训练恢复包 latest.pt 以及不可信用户上传的 pickle/PT 文件作为公开推理模型加载。
- 任意将局内隐藏手牌、模型私有调用、尚未结束的 seed、完整 replay 发给普通 spectator。
- 服务端请求在决策与提交之间状态已变化时依旧复用旧 local_index。

**唯一执行裁判**：src/ddz_core/engine.py 的 GameEngine.step()，其 Action 必须 exact membership 当前 legal_actions。模型推理返回的动作必须由权威 orchestrator 在提交前重新确认。

## 2. 当前版本已实现了什么

目前仓库**没有独立、通用、可跨进程调用的 Model Inference Microservice**。已有的是同一 Python 进程内使用 FastAPI + ArenaManager + RoleBundle 的本地 Web Arena。现有特点如下：

| 能力 | 当前状态 | 源码定位 |
|---|---|---|
| 注册模型目录 | 已实现 | ddz_registry/registry.py |
| 推理专用模型完整性校验 | 已实现 | training/checkpoints.py、registry.py |
| 多角色模型策略 | 已实现 | models/policy.py |
| 当前合法候选打分与 argmax | 已实现 | models/network.py、policy.py |
| 模型 ID + sha256 缓存 | 已实现 | web/arena.py ArenaManager._load_model |
| 三人 Arena 自动对战/暂停/单步/调速 | 已实现 | web/arena.py |
| REST 游戏生命周期 + WebSocket 快照 | 已实现 | web/api.py |
| 会话恢复与模型哈希绑定 | 已实现 | web/arena.py、persistence.py |
| 独立 /v1/decisions API | **未实现，见第 6 节设计** | 规划项 |
| 多实例推理分片、服务发现、动态批处理 | **未实现** | 规划项 |
| 通用公网级多租户安全认证/授权 | **未实现** | 当前仅 operator-local 模式 |

当前 Arena 使用 CPU 线程执行 policy.act，避免在 WebSocket 事件循环上直接进行网络评分；一个游戏一把 asyncio.Lock 串行化 step_once。游戏启动时加载并**固定**三席的模型版本，晋升新 Champion 不会替换进行中的局内模型。

## 3. 模型资产发布与可信加载

### 3.1 Checkpoint 类型

- **latest.pt**：训练恢复包，可能包含优化器、RNG、replay、训练状态；不应作为独立推理服务模型制品。
- **best.weights.pt**：inference_only bundle，仅包含四个角色网络相关状态和兼容性元数据。
- **history/step-*.weights.pt**：训练过程中的冻结推理快照，可用于受控对照。
- **runs/registry/bundles/<model-id>/weights.pt**：经 ModelRegistry.register 的不可变推理模型。

每个 .pt 文件配 .pt.json 元数据，含 sha256、checkpoint schema、网络参数形状、encoder version。当前 checkpoint loader 采用 torch.load(weights_only=True)，且注册路径必须位于明确授信目录内。

### 3.2 强制加载校验

~~~text
受信任本地制品
   |
   v
源路径 / 真实路径 / symlink 检查
   |
   v
.pt + .pt.json 完整性 SHA-256 一致
   |
   v
inference_only=true + checkpoint schema 对齐
   |
   v
state_dim=125 / action_dim=51 / 网络 hidden, schema 对齐
   |
   v
observation/action encoder 版本完全匹配
   |
   v
四角色网络键完整、权重 shape 严格匹配
   |
   v
RoleBundle(model_sha) 常驻 / eval 模式
~~~

**现有兼容性缺口**：checkpoint 中存储网络与 encoder 契约，但规则 RuleConfig 的 canonical SHA-256 尚未绑定模型元数据。后续独立服务必须把 rule hash、代码 revision、action catalogue version 加入 manifest 和服务启动检查。不能仅凭文件名称或模型自报字符串认定兼容。

### 3.3 模型注册、晋升与版本策略

- 现有 ModelRegistry 中已注册的 model_id 不允许就地覆盖权重。
- active 代表经 operator bootstrap/promote 的已选择模型；promotion 需要成对比较证据，且候选哈希/现任模型哈希必须匹配。
- 对每个推理请求应记录 model_id + sha256 + encoder/rule versions，而不是只记录 active 字符串。
- 新模型上线应**新请求或新对局生效**，已运行对局固定模型与哈希直至结束。
- 回滚是将新对局指向已经验证过的历史模型 ID，不能篡改其已有文件或旧局回放。

## 4. 现有进程内决策数据流

~~~mermaid
sequenceDiagram
    participant Arena as ArenaGame
    participant Env as DDZEnv
    participant Core as GameEngine
    participant Policy as RoleBundle
    participant Registry as ModelRegistry

    Arena->>Registry: 通过model_id读取已验证的bundle(开局时)
    Arena->>Env: current_observation()
    Env->>Core: observe(acting_seat) / legal_actions()
    Core-->>Env: 已脱敏观测 + 权威候选动作列表
    Env-->>Arena: PlayerObservation
    Arena->>Policy: act(obs, rng, epsilon=0)
    Policy->>Policy: encode state / candidates / role route / score_legal
    Policy-->>Arena: local_index + role + q_value
    Arena->>Env: step(local_index)
    Env->>Core: step(exact Action membership)
    Core-->>Arena: GameEvent / Transition
    Arena->>Arena: journal save + public snapshot + WebSocket publish
~~~

算法主流程：

1. 确认游戏未结束、当前行动者和 expected step；持有每局的串行化锁。
2. 从 env.current_observation() 获取本玩家可见字段和法定候选 tuple。
3. 通过 role_for_observation() 决定 bid/landlord/farmer_down/farmer_up。
4. encode_observation() 生成 125D state；逐 Action 调用 encode_action() 生成 K×51 候选张量。
5. 在 torch.inference_mode() + eval() 下调用 score_legal(state,candidates)。
6. 对 scores 检查全部 finite；epsilon=0 时取 argmax 的局部索引。
7. 通过当前 env.step(local_index) 执行；GameEngine.step 再验证集合成员资格。
8. 保存事件和可重放 journal，输出不包含隐藏手牌的 public_snapshot。

**重要**：模型输出的 q_value 是未校准数值，不是概率或准确率。API 若展示 score，应标注 raw Q score；不可将其假装为“这一手赢的概率 87%”。

## 5. 当前 Web Arena HTTP/WS 契约（已实现）

当前 FastAPI 实际路径在 src/ddz_web/api.py 中，包括：

| Method | Path | 用途 |
|---|---|---|
| GET | /api/health | 基本存活检查（status=ok） |
| GET | /api/models | 读取可选择的注册模型与 active |
| POST | /api/games | 创建三个 seat 的模型对局 |
| GET | /api/games/{game_id} | 获取脱敏公共比赛快照 |
| POST | /api/games/{game_id}/control | play、pause、step、speed |
| GET | /api/games/{game_id}/replay | 结束对局后下载 replay |
| WS | /api/games/{game_id}/events | 局内事件/快照推送 |
| GET | /api/lab/overview | 展示实验、报告、回放、模型等读视图 |
| GET/POST | /api/auth/status、/api/auth/login | operator 登录状态和登录 |

**现有 POST /api/games 请求：**

~~~json
{
  "model_ids": ["model-a", "model-b", "model-c"],
  "seed": 2026,
  "delay": 0.7
}
~~~

三个 model_id 必须已经在注册表内，或在服务明确启用 --allow-baseline 时用 @random 作开发基线。seed 可以省略；delay 取 0.05..3.0。并不代表向普通互联网用户开放随意上传模型。

**现有控制调用：**

~~~json
{
  "action": "step"
}
~~~

实际操作允许 play/pause/step/speed，其中 speed 时传 delay。禁止将这里的 play 与“出牌动作 PLAY”混淆：这是比赛播放控制命令。

API 默认绑定 localhost；非 loopback 需要 operator token 环境变量及 HTTPS 反向代理等措施。项目的用户会话与认证能力是 operator-local，不应宣称公网多租户安全。现有 /api/health 仅返回基本状态，**不是**模型加载好且可服务的 readiness probe。

## 6. 独立 Model Inference Service 设计（**尚未实现**）

未来建议拆为受信任的三个进程边界：

~~~text
[Web/Agent/User]
      |
      v
[Game Orchestrator API]
  - 身份和 seat 权限
  - 每局 GameEngine / state lock
  - authority legal_actions()
  - expected_step_index 校验
      |
      | 内网 mTLS / 进程 IPC，传当前 seat 的脱敏观测与完整合法候选
      v
[Model Scoring Service]
  - 已验证 registry 模型加载
  - encoder + 4-role router
  - K 个合法候选 Q-score
  - Decision + model_hash
      |
      v
[Game Orchestrator 再验证 Action / commit]
      |
      +-- Replay/journal/audit
      +-- Safe public spectator DTO
~~~

Game Orchestrator 是唯一有权持有三方 GameState 的组件；Model Scoring Service 只接收有限的 PlayerObservation 与权威候选，**没有读取完整游戏状态的权限**。若 Scoring Service 是多租户共享服务，严禁把私有观测记录在普通应用日志中。

### 6.1 建议内部接口（非现有 endpoint）

**POST /v1/decisions**，仅限 orchestrator 内部调用，建议请求为：

~~~json
{
  "request_id": "8c7ad91a-88af-43fe-95b5-eac6280f24f9",
  "game_id": "internal-game-id",
  "seat": 1,
  "expected_step_index": 42,
  "model_id": "champion-2026-10",
  "model_sha256": "sha256-from-trusted-registry",
  "rule_id": "classic_call_rob_2016_reference_v1",
  "observation_version": "ddz-observation-v1",
  "action_version": "ddz-action-v1",
  "candidate_fingerprint": "sha256-of-canonical-candidates",
  "observation": "trusted-serialized-player-observation",
  "legal_actions": "trusted-authoritative-action-list"
}
~~~

**说明**：示例中 observation 和 legal_actions 以说明字符串占位。正式实现应采用经过类型验证的对象/数组 schema，不接受任意字符串当结构化状态；仅可信 orchestrator 可构造该请求。另须明确定义 canonical JSON 与 fingerprint 算法，包括 key order、UTF-8、固定候选顺序和 rule id。

建议响应：

~~~json
{
  "request_id": "8c7ad91a-88af-43fe-95b5-eac6280f24f9",
  "game_id": "internal-game-id",
  "seat": 1,
  "expected_step_index": 42,
  "local_index": 0,
  "action": {
    "kind": "PASS",
    "counts": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "pattern_key": null
  },
  "role": "farmer_down",
  "raw_q_value": 0.31,
  "model_id": "champion-2026-10",
  "model_sha256": "sha256-from-trusted-registry",
  "candidate_fingerprint": "sha256-of-canonical-candidates"
}
~~~

响应是**字段形状示范**，并非声称在 step=42 时 PASS 一定合法：orchestrator 必须再次校验当前 last_play、seat、candidate_fingerprint、model hash 和 exact action membership 才能提交。不能信任第三方发来的伪造 legal_actions。实际响应可选择不返回 raw_q_value 给终端用户。

### 6.2 状态一致性与并发

- 每个游戏使用独立锁/串行状态机；一个决策读取 state_digest + step_index，再在提交时对比。
- scoring service 不提交游戏状态，避免模型调用跨服务产生两次 step。
- 同一 request_id + 同一 state fingerprint 是逻辑幂等请求；如重试仅计算同一动作，不可重复推进游戏。
- 如果 expected_step_index / game_id / candidate fingerprint 不匹配，返回冲突（建议 409），重新获取新观测，不得复用 local_index。
- 未加载匹配模型、编码版本冲突或候选为空时应 fail-closed，暂停对局并返回明确错误。
- 对可变长候选采用 batch offsets/padding mask，不允许从一局候选跨越到另一局的 action namespace。

### 6.3 建议错误码

| HTTP | 类型 | 处理 |
|---|---|---|
| 400/422 | MALFORMED_INPUT / INVALID_SCHEMA | 参数、向量维度或 JSON 非法，拒绝 |
| 401/403 | UNAUTHORIZED / FORBIDDEN | 调用方无权限或不能访问该游戏/seat |
| 404 | GAME_OR_MODEL_NOT_FOUND | 不暴露其他用户游戏的存在细节 |
| 409 | STALE_STATE / CANDIDATE_MISMATCH | 状态已变化，获取新观测 |
| 422 | ILLEGAL_ACTION / EMPTY_CANDIDATES | 不产生任何状态迁移 |
| 503 | MODEL_UNAVAILABLE / ENCODER_INCOMPATIBLE | 模型不可服务，禁止错误降级 |
| 500 | NONFINITE_Q / INTERNAL_INVARIANT | 记录诊断 ID，停止当前决策并告警 |

以上为**设计约定**，不是现有 api.py 的已实现返回协议。

## 7. 延迟与容量规划（建议，不是实测性能）

单次请求的总耗时大致为：

\[
T_{\text{total}} =
 T_{\text{observe}}+
 T_{\text{candidate}}+
 T_{\text{encode}}+
 T_{\text{score}}+
 T_{\text{verify}}+
 T_{\text{commit}}+
 T_{\text{serialize}}
\]

对 CPU 与 GPU 分别压测 K（合法候选数）的分位分布，不能只测 K=1。建议收集 p50/p95/p99，尤其是拥有多个三张、炸弹、飞机时的复杂候选枚举。**不应在没有测量的情况下承诺毫秒级 SLA。**

可实施策略：

1. 仅编码当前合法 K 个候选，不对 56,251 动作目录全量评分。
2. 将 state embedding 每请求只算一次，candidate encoder 支持批量算子。
3. 按 model_id/role/encoder version 聚类小批量任务，以短超时与最大 K/请求容量控制长尾。
4. Game Orchestrator 负责 CPU 牌局和法律判定；GPU 优先负担 Q-score，而非全状态 clone。
5. 非法输入及无限/NaN score 早拒绝，输出不包含私有 state 的结构化日志。
6. 模型冷启动与 warm pool 分离，并用 readiness probe 验证确实可加载所有四角色网络。

## 8. 部署指南：现有版本与未来架构分开

### 8.1 当前可部署的本地 Arena

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[web,train]'
ddz-arena --host 127.0.0.1 --port 8765 \
  --registry-dir runs/registry \
  --replay-dir runs/arena-replays
~~~

前提：runs/registry 中有经正式注册的推理 bundle。没有注册模型时，必须先按 docs/V6_MODEL_SELECTION.md 的注册流程引入受信任模型；或在**仅用于开发的本地模式**显式添加 --allow-baseline，并选择 @random，不能假称它是学习出来的策略。

~~~bash
# 仅供本机开发基线；不代表启用学习模型
ddz-arena --host 127.0.0.1 --port 8765 --allow-baseline
~~~

本地打开 http://127.0.0.1:8765/。在非 loopback 监听环境下，现有 CLI 要求受保护的 operator token 环境变量和 HTTPS 反向代理；直接把 8765 端口暴露公网不是合规部署方案。

~~~bash
# 读取当前 CLI 实际支持的全部安全/目录参数
ddz-arena --help
~~~

### 8.2 建议未来服务交付形态（**未实现**）

~~~text
deploy/
  inference/
    Dockerfile
    compose.yaml
    config.example.yaml
    healthcheck.py
  proxy/
    reverse-proxy.example.conf
~~~

计划中需要完善：独立服务启动参数、只读模型文件系统、最小权限用户、CPU/GPU 设备选择、内网鉴权/网络分段、部署前模型 hash 检查、资源限额、请求超时、监控和灰度。**不要把这份规划目录当作仓库现有文件**。

## 9. 安全和隔离清单

- **模型供应链**：不接受公网任意文件路径、URL 或 PT 上传；加载前检查哈希、可信来源、四角色参数形状。
- **鉴权授权**：仅 Game Orchestrator 能提交动作；面向玩家的读取只能获取自己座位可见状态；观众只能看公共视图。
- **时间边界**：未结束前不得泄露种子或可逆完整 replay，不能把审计存档作为实时观众的 JSON。
- **日志最小化**：允许 request_id、seat、role、model SHA 缩写、耗时、候选数；不默认打印 own_physical_cards 或完整私有观测向量。
- **输入验证**：限制张数、数组大小、维度、model_id 字符集、有限浮点值；拒绝负索引、bool 冒充 int、重复或失序候选。
- **竞态**：每局 step 原子提交。迟到的模型响应必须被丢弃，不能对更新后的手牌继续生效。
- **失败语义**：模型不兼容、timeout、失去规则裁判时暂停/错误，不擅自 PASS、随机 PLAY 或判负。
- **发布**：模型晋升必须具备规则合法性测试、端到端 replay、统计对照和可回滚模型 ID。

## 10. 可观测性、SLO 与追责性设计

建议指标（具体门槛先测基线再制定）：

| 类别 | 核心字段 |
|---|---|
| 合法性 | illegal_proposed_total、illegal_executed_total（必须为零）、stale_rejections |
| 可靠性 | failed_decisions、NaN/Inf、checkpoint_mismatch、timeout、replay_digest_mismatch |
| 延迟 | candidate_generation_p95、encoding_p95、model_score_p95、commit_p95、end_to_end_p99 |
| 负载 | requests_per_second、concurrent_games、candidate_count_distribution、GPU_memory |
| 模型 | model_id、model_sha256、rule_hash、encoder_version、role、stage |
| 质量 | landlord_win_rate、farmer_win_rate、balanced_score、truncated_rate、paired_ci |

ModelRegistry 晋升不代表推理服务通过全部端到端验收；应在 shadow/canary 中与 incumbent 使用相同局面做**合法性和延迟**比对，再切换新对局模型绑定。不能从 shadow 输出直接推进真实 Arena 状态。

## 11. 端到端验收（建议的 CI/TestGate）

| 用例 | 输入条件 | 必须结果 |
|---|---|---|
| 法律性 | 上家对 2，本家有对 Q | 对 Q 不在候选中，不可执行 |
| 类型 | local_index=-1、bool、float、越界 | 拒绝，不改状态 |
| 歧义 | 相同 counts 的不同 pattern_key | 动作保持区分，提交与 replay 一致 |
| 角色 | 地主 seat=0/1/2 | farmer_up/down 正确按相对座位路由 |
| 无法压制 | 仅能 PASS 的跟牌局面 | 模型返回 PASS；领出状态不存在 PASS |
| 空候选 | 终局/错误阶段 | 不调用 scorer、不推进状态 |
| 数值 | NaN / Inf Q | fail-closed；不输出随机非法动作 |
| 漂移 | 换 rule hash / encoder / hidden | 启动或请求时拒绝不兼容模型 |
| 并发 | 两个相同游戏 step 并发 | 至多一次提交；另一请求 stale/retry |
| 恢复 | 相同 seed、Action 序列、哈希 bundle | replay digest 相同 |
| 脱敏 | spectator、其他 seat 请求 | 不可看到隐藏牌和完整未完局 seed |
| 回滚 | 晋升期间已有比赛继续进行 | 旧局模型绑定不变，新局可选新模型 |

**已有可运行测试入口：**

~~~bash
python -m pip install -e '.[dev,train,web,webtest]'
pytest -q tests/test_training.py tests/test_v6_registry.py \
  tests/test_v7_arena.py tests/test_v11_reliability.py \
  tests/test_engine.py
pytest -q
~~~

建议补充独立 scorer 的契约测试与多进程/服务重试测试；目前不存在 /v1/decisions endpoint，因此不能声称相应服务测试已经通过。

## 12. 建议实施顺序

1. **S0 — Legal-first**：在环境、训练、推理、Arena 加相同 golden legality test 与非法动作为零发布门禁。
2. **S1 — Bundle contract**：补 canonical rule hash、代码 revision、模型 manifest，完成加载拒绝与 train/serve parity。
3. **S2 — Scoring-only 内部接口**：受信任 orchestrator 构造 PlayerObservation + 权威动作，评分服务返回动作与版本指纹。
4. **S3 — Game Orchestrator 原子提交**：expected state、请求幂等、并发冲突、审计 replay 与安全 DTO。
5. **S4 — 性能治理**：动态 K 候选 microbatch、warm pool、并发压测、故障注入。
6. **S5 — 运营质量与回滚**：受控灰度、冻结对手锦标赛验证、模型版本 pin、对局后切换、告警。

## 13. 关联文档

- **规则裁判**：docs/GAME_RULES.md、docs/RULE_SPEC.md
- **权威状态/环境/动作契约**：docs/ENVIRONMENT_DESIGN.md、docs/V3_ENV_CONTRACT.md、docs/DATA_CONTRACTS.md
- **模型与训练**：docs/MODEL_TRAINING.md、docs/V4_TRAINING.md、docs/V5_TRAINING_AND_TOURNAMENT.md
- **模型选择与注册**：docs/V6_MODEL_SELECTION.md
- **现有 Web Arena**：docs/V7_WEB_ARENA.md
- **可靠性与训练质量**：docs/V11_RECOVERY_AND_EVALUATION.md、docs/V12_TRAINING_QUALITY.md


## Integration note

The merged code supports versioned v1 and v2 candidate-action encoders. An opt-in local diagnostic endpoint can display full hands for debugging; standard public snapshots and model observations remain redacted. All executed actions continue to be validated by GameEngine.legal_actions().
