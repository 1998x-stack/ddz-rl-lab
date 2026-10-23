# DDZ RL Lab · 斗地主多智能体强化学习实验室

[![CI](https://github.com/1998x-stack/ddz-rl-lab/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/1998x-stack/ddz-rl-lab/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Stage](https://img.shields.io/badge/stage-P0%20%2B%20P1-orange)
![MIT](https://img.shields.io/badge/license-MIT-green)

> **独立、无投注的斗地主强化学习研究工程**。第一阶段提供可执行的经典三人斗地主规则引擎、完整的合法动作、按玩家隔离的 Observation、确定性回放、Loguru 日志和测试。后续版本计划逐步增加 PettingZoo ENV、DMC、自博弈训练、Checkpoint 选拔以及三 Agent Web Arena。
>
> **非腾讯官方项目**：不包含腾讯客户端、素材、专有协议、用户数据、虚拟货币、充值及真实资金结算；不能将本仓库的规则实现认定为腾讯微信小游戏当前全部玩法的逐项复刻。

## 项目状态

| 里程碑 | Code Category | 当前状态 | 内容 |
| --- | --- | --- | --- |
| P0 | `specs` | ✅ 已实现 | 规则规格、数据契约、设计说明、Code Category 路线图 |
| P1 | `game_core` | ✅ 已实现 | 牌、牌型、叫抢地主、合法动作、完整对局、回放、Loguru |
| P2 | `game_env` | 规划中 | PettingZoo AEC、Observation Encoder、Reward、Vector ENV |
| P3 | `rl_models` | 规划中 | Role-Aware Policy、MLP/LSTM、合法候选动作评分 |
| P4 | `rl_training` | 规划中 | DouZero-style DMC、Actor/Learner、Self-Play |
| P5 | `rl_evaluation` | 规划中 | 固定对手池、Tournament、Checkpoint 选拔 |
| P6 | `inference_server` | 规划中 | Model Registry、FastAPI、WebSocket、Agent 调度 |
| P7 | `web_console` | 规划中 | 实时三 Agent Arena、回放、训练面板 |
| P8 | `observability` | 规划中 | 跨进程日志收集、Metrics、Tracing、部署 |

**特别说明：本仓库当前还没有强化学习模型、checkpoint、Web 服务器或浏览器牌桌。** `ddz-demo` 使用随机合法动作/简单规则型基线策略进行自动对局，不等于训练完成的 RL 系统。

## 快速开始

最低环境：**Python 3.10+**、CPU、可安装 Python 依赖的网络（安装后无需联网）。不需要 GPU、CUDA、Redis、数据库或 Docker。

### Ubuntu 22.04 / 24.04

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip

git clone https://github.com/1998x-stack/ddz-rl-lab.git
cd ddz-rl-lab
# 如要固定到第一阶段，而不是使用最新 main:
# git switch v1.0

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q
```

### macOS

```bash
git clone https://github.com/1998x-stack/ddz-rl-lab.git
cd ddz-rl-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

Ubuntu 20.04 的系统 Python 可能早于 3.10。应安装独立 Python 3.10+ 环境，**不要覆盖系统 Python**。纯运行环境可以选择 `python -m pip install -e .`，无须安装测试依赖。

## 一键演示与回放

```bash
ddz-demo --games 3 --seed 2026 \
  --replay-out replays/example.json \
  --log-dir logs --log-level INFO

# 不依赖可执行脚本入口的替代命令
python -m ddz_core.cli --games 5 --seed 42
```

示例会输出各局的 seed、地主座位、胜方、决策次数、发牌次数和状态哈希。最后一局同时导出 Replay JSON 并自动验证回放状态摘要。

手动验证 Replay：

```python
import json
from ddz_core import replay

with open("replays/example.json", encoding="utf-8") as file:
    record = json.load(file)

engine = replay(record)
assert engine.state_digest() == record["state_digest"]
print(record["winner"], engine.state.step_index)
```

**Replay 文件是受控研究资产**：记录 seed、规则配置和动作序列，可重建未公开手牌；普通实时观众不能直接访问原始 Replay 数据。

## 代码目录

```text
ddz-rl-lab/
├── .github/workflows/ci.yml      # GitHub Actions 测试矩阵
├── docs/
│   ├── ROADMAP.md                # P0–P8 细分与验收门禁
│   ├── RULE_SPEC.md              # 游戏规则约定和规则分歧
│   ├── DATA_CONTRACTS.md         # 数据结构和序列化契约
│   ├── DESIGN_NOTES.md           # 规则内核、隔离和可重放性设计
│   └── VERIFICATION.md           # 本阶段测试记录
├── examples/run_baseline.sh      # 基线测试演示
├── src/ddz_core/
│   ├── cards.py                  # 54张物理牌ID和15维点数编码
│   ├── actions.py                # Action、Pattern、Kind 枚举
│   ├── rules.py                  # 牌型、合法候选动作、比较器
│   ├── state.py                  # 内部状态、玩家观察、游戏事件
│   ├── engine.py                 # 叫抢、出牌、状态转移、事件回放
│   ├── logging_config.py         # Loguru 配置
│   ├── cli.py                    # 无资金随机基线自动对局
│   └── __init__.py
├── tests/
│   ├── test_cards.py
│   ├── test_patterns.py
│   ├── test_engine.py
│   └── test_fuzz.py
├── pyproject.toml
├── CHANGELOG.md
└── LICENSE
```

分层依赖原则：`cards / actions / rules` → `state / engine` → `CLI / future ENV adapter`。**`game_core` 不得依赖 PyTorch、PettingZoo 或 Web 服务器。**

## 规则与游戏状态机

本阶段的规则依据公开的经典三人斗地主玩法，并明确冻结以下项目自有的约定：

1. 54 张不同的物理牌；3 人各发 17 张，保留 3 张底牌。出牌方向按座位 `0 → 1 → 2 → 0`。
2. `CALL / DECLINE_CALL`，所有玩家均未叫地主则以确定性规则重新发牌；第一次叫地主后进入 `ROB / DECLINE_ROB` 阶段。
3. 抢地主资格、顺序和限制在 `docs/RULE_SPEC.md` 中明确，不声称和腾讯所有版本相同。
4. 确定地主后公开底牌，地主手牌变为 20 张；地主先出，其余两家可以压牌或 PASS。
5. 连续两家 PASS 后，最后成功出牌的玩家重新领牌；地主先出完则地主胜，任意农民先出完则农民阵营胜。
6. 全员不叫可能多次重新发牌。超过 `RuleConfig.max_redeals` 则 `truncated=True`、`winner=None`，不能当成正常胜负。

支持：单张、对子、三张、三带一、三带二、顺子、连对、飞机、飞机带单、飞机带对、四带二单、四带两对、炸弹和王炸。合法动作以 **15维点数计数 + 牌型解释 `pattern_key`** 标识，解决同一组牌可能存在多种语义解释的问题。

一些有争议的规则（如飞机单翼重复点数、四带二单牌重复、王是否作为附件牌）通过 `RuleConfig` 明确选择，不应不经版本管理就修改。详见 [RuleSpec](docs/RULE_SPEC.md)。

## 数据结构和 API

| 结构 | 作用 | 信息权限 |
| --- | --- | --- |
| `GameState` | 三手完整牌、所有内部状态、种子、事件 | 仅可信规则引擎 |
| `PlayerObservation` | 自己的牌、公开底牌、剩余牌数、历史和合法动作 | 仅对应座位 |
| `Action` | 叫地主、抢地主、出牌、PASS 等动作 | 当前行动人 |
| `Pattern` | 牌型主体、连牌长度和总牌数 | 按可见信息使用 |
| `GameEvent` | 可按顺序回放的状态变化 | 对外需做隐私过滤 |
| `RuleConfig` | 游戏规则版本与可配置差异 | 可公开 |

完整研究用法：

```python
import random
from ddz_core import GameEngine
from ddz_core.cli import baseline_move
from ddz_core.state import Phase

engine = GameEngine(seed=123)
rng = random.Random(123)

while engine.state.phase != Phase.FINISHED:
    seat = engine.state.current_seat
    obs = engine.observe(seat)
    action = baseline_move(engine, rng)
    assert action in obs.legal_actions
    engine.step(action)

record = engine.export_replay()
print(record["winner"], engine.state_digest())
```

请注意：示例直接读取 `engine.state.current_seat` 只是**可信模拟器内的调度**。未来 RL Agent 只能接收 `engine.observe(seat)`，**不得读取完整 `GameState`**，否则会泄漏其他两家隐藏牌，破坏强化学习问题定义。

## 测试、验收与 GitHub Actions

```bash
python -m pytest -q
python -m ddz_core.cli --games 3 --seed 2026 \
  --replay-out /tmp/ddz-replay.json --log-dir /tmp/ddz-logs
```

首版包含 **43 项 pytest 测试**，涵盖：

- 洗牌、牌唯一性、数量和15维编码；
- 所有基础牌型、复合牌型、无效组合和炸弹比较；
- 叫牌/抢地主、全不叫重新发牌、结束与异常截断；
- 连续两次 PASS 后的重新领牌；
- 非法动作不修改状态、玩家隐藏信息隔离；
- 固定 seed、完整事件序列、状态哈希和 Replay；
- 多种随机初始牌局的牌守恒及完整结束。

本地验证已执行 43 项测试以及三局基线对局。仓库 CI 为 Python **3.10、3.12、3.13** 的矩阵；GitHub 上的最终 CI 结果请以 [Actions 页面](https://github.com/1998x-stack/ddz-rl-lab/actions) 为准，不能把本地验证冒充云端已经通过。

43 项测试并不等价于对所有版本的腾讯规则或所有可组合手牌的形式化证明；P2 还应扩充多 seed、Property-Based、参考规则差分和隐藏信息测试。

## Loguru 日志体系

使用 [Loguru](https://github.com/Delgan/loguru) 记录游戏创建、终局、Replay 输出与异常。示例：

```bash
ddz-demo --games 1 --seed 42 --log-level DEBUG --log-dir logs
```

正式系统需要区分三类数据：**Loguru 应用日志**、**GameEvent 可重放事件**、**训练 Metrics**。Loguru 不承担游戏权威存储；大量训练时不能逐决策同步刷盘，应使用日志分级、轮转、保留和集中采集。

## 分支与发布策略

- [`v1.0`](https://github.com/1998x-stack/ddz-rl-lab/tree/v1.0)：**P0 + P1 第一阶段可运行代码**，作为可回溯的稳定基线。
- [`v2.0`](https://github.com/1998x-stack/ddz-rl-lab/tree/v2.0)：从 `v1.0` 派生，作为 P2 多智能体 ENV 的开发入口。仅创建此分支**不代表已完成 P2 的代码**。
- [`main`](https://github.com/1998x-stack/ddz-rl-lab/tree/main)：同步到**最新已经过测试的里程碑分支**。新增功能在分支中开发并通过测试后，再以快进方式更新 main；不强制覆盖已有无关改动。
- 未来 `v3.0`、`v4.0` 等遵循同一原则。

```bash
git fetch --all
git switch v1.0      # 第一阶段基线
git switch v2.0      # 第二阶段开发入口
git switch main      # 最新经验证分支
```

这里的 `v1.0`、`v2.0` 是 Git 分支名；当前 Python Package Version 是 `0.1.0`，二者不是同一个版本字段。

## 常见问题

**Q：为什么没有三 Agent 网页？** 当前只交付规则引擎和 CLI 基线。完整 RL 自博弈、checkpoint 和在线 Web Arena 是后续阶段。

**Q：必须有 GPU 吗？** 不需要。P1 完全可在 CPU 上运行。训练阶段可单独加入 GPU 和分布式 Worker。

**Q：我能重新生成某局吗？** 可以固定 seed、规则配置和动作序列；`replay(record)` 会重放并比较最终状态摘要。模型重新采样可能涉及不同的探索 RNG，不应与“原动作回放”混淆。

**Q：如何调整飞机附件规则？** 修改并版本化 `RuleConfig` 和 RuleSpec；之后训练的数据、模型也需显式绑定规则/编码器版本。

**Q：和 RLCard / DouZero 是同一套源码吗？** 不是。规则核心为独立编写，可将这些学术项目作为算法和差分测试参考。

## 许可证与项目边界

源码采用 [MIT License](LICENSE)。本项目只用于无资金的规则模拟、教学和 AI 研究；不提供押注、充值、可兑换筹码、第三方线上作弊或未授权游戏平台接入功能。

参考：[DouZero](https://github.com/kwai/DouZero) · [RLCard](https://github.com/datamllab/rlcard) · [PettingZoo](https://pettingzoo.farama.org/)。
