# 基于改进型多智能体强化学习的无人机协同区域覆盖与目标搜寻

> **UAV Cooperative Area Coverage and Target Search Based on Improved Multi-Agent Reinforcement Learning**

[![Python](https://img.shields.io/badge/Python-%E2%89%A5%203.9-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A5%202.0-red?logo=pytorch)](https://pytorch.org/)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-1.24.3-green)](https://pettingzoo.farama.org/)
[![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/HTML)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 目录结构](#2-目录结构)
- [3. 环境依赖与安装](#3-环境依赖与安装)
- [4. 快速开始](#4-快速开始)
- [5. 算法说明](#5-算法说明)
- [6. 实验设计](#6-实验设计)
- [7. 评价指标](#7-评价指标)
- [8. 配置说明](#8-配置说明)
- [9. Web 训练控制台](#9-web-训练控制台)
- [10. 测试](#10-测试)
- [11. GitHub 上传前检查](#11-github-上传前检查)
- [12. 参考文献](#12-参考文献)
- [13. 项目信息](#13-项目信息)

---

## 1. 项目概述

### 1.1 研究问题

本项目以多架无人机在灾害救援、区域巡逻、目标侦察等场景中的协同搜索任务为背景，将无人机集群协作问题抽象为多智能体强化学习（Multi-Agent Reinforcement Learning, MARL）问题。

**核心命题：让多个智能体从最初的无序移动，逐渐学习形成分工协作行为，使不同无人机能够自动分散到不同目标区域，提高搜索覆盖率，降低碰撞风险，并提升整体任务完成效率。**

### 1.2 技术路线

```
随机策略（基线）
    │
    ▼
独立学习 IDDPG（对比方法）
    │
    ▼
原始 MADDPG（基准方法，CTDE 范式）
    │
    ▼
RS-MADDPG（改进方法）
    ├── 目标分配引导奖励
    ├── 冗余覆盖惩罚
    ├── 安全距离约束
    └── 动态奖励权重机制
```

### 1.3 关键特性

- **4 种算法完整对比**：Random / IDDPG / MADDPG / RS-MADDPG 逐层递进
- **奖励塑形改进方案**：在 MADDPG 框架上针对协同覆盖任务的核心痛点（重复覆盖、碰撞风险）精准优化奖励函数
- **消融实验验证**：逐模块验证目标分配、冗余惩罚、安全距离、动态权重的独立贡献
- **泛化性测试**：N=3/4/5 不同规模场景下的算法扩展性评估
- **工业风 Web 训练控制台**：纯 HTML/CSS/JS 前端 + Python 标准库 HTTP 服务器，Canvas 实时 2D 渲染 + SSE 事件流推送，零外部框架依赖
- **实验模式与自动总结**：支持单次训练、随机 vs 训练后、算法对比、消融实验、泛化实验；实验结束后自动生成结构化指标总结和报告文本
- **DeepSeek 报告生成**：可选调用 DeepSeek，将结构化指标转换为课程报告风格的自然语言结论
- **统一训练器**：支持 Ctrl+C 中断恢复，自动保存 checkpoint 和日志
- **完整测试覆盖**：单元测试覆盖 agents、env、networks、rewards、utils 等核心模块

### 1.4 环境平台

基于 [PettingZoo](https://pettingzoo.farama.org/) MPE 环境中的 `simple_spread_v3` 任务构建，将智能体覆盖目标点映射为无人机协同区域搜索场景。

| 组件 | 版本/说明 |
|------|----------|
| Python | ≥ 3.9 |
| PyTorch | ≥ 2.0 |
| PettingZoo | 1.24.3 |
| Gymnasium | ≥ 0.29 |
| NumPy | ≥ 1.24 |
| Matplotlib | ≥ 3.7 |
| PyYAML | ≥ 6.0 |
| Pytest | ≥ 7.0 |
| 前端 | 纯 HTML5 / CSS3 / ES6（无需 Node.js / npm / 构建工具） |
| 服务器 | Python 标准库 `http.server` + `threading`（无需 FastAPI / Flask） |
| 硬件 | CPU 即可训练（3 智能体任务规模较小） |

---

## 2. 目录结构

```
uav_marl_project/
├── web_server.py                       # ★ Web 训练控制台服务器（纯 Python 标准库）
├── web_static/                         # ★ Web 前端静态资源
│   ├── index.html                       #   单页应用（工业风暗色 UI）
│   ├── app.js                           #   Canvas 2D 渲染 + SSE 事件流 + 训练曲线
│   └── styles.css                       #   暗色主题样式表
├── app.py                              # Streamlit 训练前端（旧版，保留备用）
├── pages/                              # Streamlit 多页面（旧版）
│   ├── 1_📊_实验一_随机vs训练.py
│   ├── 2_📊_实验二_算法对比.py
│   ├── 3_📊_实验三_消融实验.py
│   ├── 4_📊_实验四_泛化实验.py
│   └── 5_综合汇报面板.py
├── env/                                # 环境封装
│   ├── __init__.py
│   └── simple_spread_wrapper.py         # simple_spread_v3 封装（含自定义奖励）
├── agents/                             # 智能体实现
│   ├── __init__.py
│   ├── base_agent.py                    # 智能体基类（save/load 接口）
│   ├── random_agent.py                  # 随机策略
│   ├── iddpg_agent.py                   # 独立 DDPG
│   ├── maddpg_agent.py                  # 原始 MADDPG（含 TD3 稳定性增强）
│   └── rs_maddpg_agent.py               # RS-MADDPG（奖励塑形改进版）
├── rewards/                            # 奖励塑形模块
│   ├── __init__.py
│   ├── base_reward.py                   # 原始环境奖励提取
│   ├── assignment.py                    # 目标分配引导奖励（含迟滞机制）
│   ├── redundancy.py                    # 冗余覆盖惩罚
│   ├── safety.py                        # 安全距离约束
│   └── weight_scheduler.py              # 动态权重调度器
├── networks/                           # 神经网络
│   ├── __init__.py
│   ├── actor.py                         # Actor 网络（FC+ReLU+Sigmoid）
│   └── critic.py                        # Critic 网络（含截断双 Q）
├── frontend/                           # 可视化组件（供 Streamlit 和命令行脚本使用）
│   ├── __init__.py
│   ├── components.py                    # 配色常量、可复用 UI 组件
│   ├── render_2d.py                     # 2D 仿真区域渲染（pygame/matplotlib）
│   └── charts.py                        # 训练曲线 / 柱状图 / 轨迹图（plotly）
├── reports/                            # 实验结果总结与报告生成
│   ├── __init__.py
│   └── report_generator.py              # 结构化指标统计、诊断、建议、报告文本
├── utils/                              # 工具模块
│   ├── __init__.py
│   ├── config_loader.py                 # YAML 配置加载器
│   ├── replay_buffer.py                 # 经验回放池（支持 Tensor）
│   ├── ou_noise.py                      # Ornstein-Uhlenbeck 探索噪声
│   └── metrics.py                       # 评价指标计算
├── experiments/                        # 实验脚本
│   ├── __init__.py
│   ├── adapters.py                      # 智能体适配器（IDDPG/Random → Trainer 接口）
│   ├── trainer.py                       # 通用训练器（支持中断恢复）
│   ├── exp1_random_vs_trained.py        # 实验一：随机 vs 训练后策略
│   ├── exp2_algorithm_comparison.py     # 实验二：四算法横向对比
│   ├── exp3_ablation.py                 # 实验三：消融实验
│   ├── exp4_generalization.py           # 实验四：泛化实验
│   └── export_report.py                 # 报告导出工具
├── configs/                            # 配置文件
│   └── default.yaml                     # 统一配置文件（环境/网络/训练/奖励）
├── tests/                              # 单元测试
│   ├── __init__.py
│   ├── test_agents.py                   # 智能体测试
│   ├── test_env_wrapper.py              # 环境封装测试
│   ├── test_networks.py                 # 网络模块测试
│   ├── test_rewards.py                  # 奖励模块测试
│   ├── test_rs_maddpg.py               # RS-MADDPG 集成测试
│   └── test_utils.py                    # 工具模块测试
├── outputs/                            # 输出目录（自动创建）
│   ├── models/                          # 模型权重
│   ├── logs/                            # 训练日志（JSON 格式）
│   └── exports/                         # 导出图片
├── .env.example                         # DeepSeek 配置示例（可复制为 .env.local）
├── .gitignore                           # 忽略密钥、缓存、训练产物和 W&B 日志
├── start_uav_marl.command               # macOS 一键启动脚本
├── start_uav_marl.bat                   # Windows 一键启动脚本
├── LICENSE                              # MIT License
└── requirements.txt                    # 依赖清单
```

---

## 3. 环境依赖与安装

### 3.1 环境要求

- Python ≥ 3.9
- PyTorch ≥ 2.0
- 操作系统：macOS / Linux / Windows

### 3.2 安装步骤

```bash
# 1. 进入项目目录
cd uav_marl_project

# 2. 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 3. 安装核心依赖
pip install -r requirements.txt
```

### 3.3 依赖清单

| 包名 | 版本 | 用途 | 必需 |
|------|------|------|:----:|
| `pettingzoo` | 1.24.3 | 多智能体环境（MPE simple_spread_v3） | ✓ |
| `torch` | ≥ 2.0 | 深度学习框架 | ✓ |
| `gymnasium` | ≥ 0.29 | 强化学习环境接口 | ✓ |
| `numpy` | ≥ 1.24 | 数值计算 | ✓ |
| `matplotlib` | ≥ 3.7 | 训练曲线、轨迹图渲染 | ✓ |
| `pygame` | ≥ 2.5 | PettingZoo MPE 环境渲染引擎 | ✓ |
| `pyyaml` | ≥ 6.0 | YAML 配置文件解析 | ✓ |
| `pytest` | ≥ 7.0 | 单元测试 | ✓ |
| `streamlit` | ≥ 1.28 | Streamlit 训练前端（旧版备用） | |
| `plotly` | ≥ 5.17 | 交互式图表（Streamlit / 实验脚本） | |
| `kaleido` | ≥ 0.2.1 | Plotly 静态图片导出 | |
| `supersuit` | ≥ 3.9 | 环境向量化 | |
| `opencv-python-headless` | ≥ 4.8 | 图像处理（无头模式） | |
| `imageio` | ≥ 2.31 | GIF 动画导出 | |
| `wandb` | ≥ 0.16 | 可选：实验跟踪与日志 | |

> **Web 训练控制台**本身仅依赖 Python 标准库（`http.server`, `threading`, `json`），前端使用纯 HTML/CSS/JS，无需任何外部框架。W&B 和 DeepSeek 均为可选能力：W&B 使用 `wandb` 包；DeepSeek 通过标准库 `urllib` 调用 HTTP API，不需要额外 SDK。

### 3.4 本机已有虚拟环境

如果使用当前项目机器上的现有环境，可直接进入项目并使用：

```bash
cd /Users/wayne/项目/曹佳宁/uav_marl_project
/Users/wayne/环境/uav_marl_venv/bin/python web_server.py --port 8600
```

如果希望在当前终端激活该环境：

```bash
source /Users/wayne/环境/uav_marl_venv/bin/activate
python web_server.py --port 8600
```

---

## 4. 快速开始

### 4.1 启动 Web 训练控制台（推荐）

#### 方法一：双击一键启动脚本（macOS 推荐）

直接双击项目根目录下的：

```text
start_uav_marl.command
```

脚本会自动完成：

1. 检测已有虚拟环境 `/Users/wayne/环境/uav_marl_venv`，也可用 `UAV_MARL_VENV` 覆盖路径
2. 如果没有检测到，则创建项目本地 `.venv`
3. 安装 / 更新 `requirements.txt` 依赖
4. 检测 `DEEPSEEK_API_KEY`
5. 如果没有 API Key，提示用户输入，并保存到 `.env.local`
6. 检查 `8600` 端口是否被占用
7. 自动启动 Web 服务并打开浏览器

> `.env.local` 会被 `.gitignore` 忽略，不会被提交。该文件权限会设置为仅当前用户可读写。

#### 方法二：命令行启动

```bash
python web_server.py --port 8600
```

浏览器打开 `http://127.0.0.1:8600`，即可看到工业风训练控制台：

- **左侧边栏**：配置实验模式、算法、智能体数量、超参数、安全与覆盖参数、W&B 追踪
- **右侧主区域**：KPI 指标卡片（Episode / 覆盖率 / 碰撞 / 奖励 / 完成步数）、Canvas 2D 实时仿真动画、训练曲线图、事件流日志
- **结果总结区**：实验结束后自动展示结构化结论、指标总览表、诊断建议和报告文本
- **一键操作**：点击「开始训练」启动后台训练线程，点击「停止」中断训练
- **实时推送**：通过 Server-Sent Events (SSE) 将每个环境步的智能体位置实时推送到浏览器，Canvas 逐帧渲染运动轨迹

```
┌──────────────────────────┬────────────────────────────────────────┐
│  侧边栏 (330px)          │  主工作区                                │
│                          │  ┌──────────────────────────────────┐  │
│  UAV MARL                │  │  Episode / 覆盖率 / 碰撞 / 奖励   │  │
│  协同覆盖训练控制台       │  │  完成步数 KPI 卡片               │  │
│                          │  └──────────────────────────────────┘  │
│  ┌─ 实验配置 ─────────┐  │  ┌──────────────────┬─────────────┐  │
│  │ 模式 [单次训练]     │  │  │                  │  训练曲线    │  │
│  │ 算法 [RS-MADDPG]   │  │  │                  │              │  │
│  │ 智能体 [3]         │  │  │  Canvas 2D       │              │  │
│  │ Episodes [500]     │  │  │  实时仿真        │  ┌────────┐  │  │
│  │ Batch [256]        │  │  │                  │  │ 事件流  │  │  │
│  └────────────────────┘  │  │                  │  └────────┘  │  │
│  ┌─ 训练参数 ─────────┐  │  └──────────────────┴─────────────┘  │
│  │ Actor LR [0.001]   │  │                                        │
│  │ Critic LR [0.001]  │  │                                        │
│  │ Gamma [0.95]       │  │                                        │
│  └────────────────────┘  │                                        │
│  ┌─ 安全与覆盖 ───────┐  │                                        │
│  │ 覆盖半径 [0.12]    │  │                                        │
│  │ 安全距离 [0.10]    │  │                                        │
│  │ ☑ 动态奖励权重     │  │                                        │
│  └────────────────────┘  │                                        │
│                          │                                        │
│  [开始训练] [停止]       │                                        │
│                          │  ┌──────────────────────────────────┐  │
│                          │  │  结果总结 / 指标表 / 报告文本     │  │
│                          │  └──────────────────────────────────┘  │
└──────────────────────────┴────────────────────────────────────────┘
```

### 4.2 可选：配置 W&B 与 DeepSeek

#### W&B 实验追踪

W&B 可在页面左侧「实验追踪」中勾选启用。首次使用前请先登录：

```bash
wandb login
```

启用后，每个 episode 会记录：

- `reward/episode`
- `task/coverage_rate`
- `safety/collision_count`
- `task/avg_min_distance`
- `task/redundancy_rate`
- `task/completion_steps`
- `loss/critic`
- `loss/actor`
- `episode/steps`
- `episode/total_steps`

#### DeepSeek 自动报告

DeepSeek 用于将结构化指标总结转换成自然语言课程报告。启动服务前设置环境变量：

```bash
export DEEPSEEK_API_KEY="你的 DeepSeek API Key"
python web_server.py --port 8600
```

如果使用当前本机虚拟环境：

```bash
export DEEPSEEK_API_KEY="你的 DeepSeek API Key"
/Users/wayne/环境/uav_marl_venv/bin/python web_server.py --port 8600
```

页面中点击「DeepSeek 生成报告」即可生成自然语言结论。若未设置 `DEEPSEEK_API_KEY`，训练和结构化总结仍可正常使用，只是 LLM 报告按钮会提示缺少密钥。

### 4.3 命令行训练

Web 控制台的训练逻辑与命令行实验脚本共享同一套智能体和环境模块，你也可以跳过 Web UI 直接运行实验脚本。

#### 实验一：随机策略 vs 训练后策略对比

```bash
# 快速评估（使用已有模型）
python experiments/exp1_random_vs_trained.py

# 指定智能体数量
python experiments/exp1_random_vs_trained.py --num_agents 4

# 若无预训练模型，自动快速训练 2000 episodes
python experiments/exp1_random_vs_trained.py --quick_train 2000
```

#### 实验二：四算法性能对比

```bash
# 快速验证（5000 episodes，1 seed）
python experiments/exp2_algorithm_comparison.py

# 完整实验（20000 episodes × 3 seeds）
python experiments/exp2_algorithm_comparison.py --full

# 仅训练指定算法
python experiments/exp2_algorithm_comparison.py --algorithms maddpg,rs_maddpg

# 自定义参数
python experiments/exp2_algorithm_comparison.py --episodes 10000 --seeds 42,123
```

#### 实验三：消融实验

```bash
# 快速验证
python experiments/exp3_ablation.py

# 自定义参数
python experiments/exp3_ablation.py --episodes 10000 --seeds 42,123

# 仅训练指定消融组
python experiments/exp3_ablation.py --groups maddpg,assign,full
```

#### 实验四：泛化实验

```bash
# 默认 N=3/4/5
python experiments/exp4_generalization.py

# 仅测试 N=3,4
python experiments/exp4_generalization.py --scales 3,4

# 仅测试 MADDPG
python experiments/exp4_generalization.py --algorithms maddpg
```

### 4.4 查看实验结果

所有实验数据自动保存到 `outputs/logs/` 和 `outputs/models/` 目录：

```bash
# 训练日志（JSON 格式，可直接用 pandas 读取分析）
ls outputs/logs/

# 模型权重（.pt 文件）
ls outputs/models/
```

训练曲线和评估结果也会在 Web 控制台的 Canvas 图表中实时呈现。

### 4.5 运行测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行指定测试模块
pytest tests/test_rewards.py -v
pytest tests/test_rs_maddpg.py -v
```

---

## 5. 算法说明

### 5.1 算法概览

| 算法 | 定位 | 核心特点 |
|------|------|---------|
| **Random** | 基线对照 | 均匀随机采样动作，不学习 |
| **IDDPG** | 对比方法 | 独立 DDPG，各智能体独立学习，面临非平稳性问题 |
| **MADDPG** | 基准方法 | 集中式训练 + 分布式执行（CTDE），集中式 Critic 使用全局信息 |
| **RS-MADDPG** | 改进方法 | 在 MADDPG 基础上增加四项奖励塑形改进 |

### 5.2 Random（随机策略）

每个智能体从 5 维连续动作空间中均匀随机采样。用于说明"不学习就无法协同"——智能体无序随机游走，覆盖率低，碰撞频繁。

**实现位置**：[`agents/random_agent.py`](agents/random_agent.py)

### 5.3 IDDPG（独立 DDPG）

每个智能体独立维护自己的 Actor 和 Critic 网络，Critic 仅使用自身局部观测和动作。将其他智能体视为环境的一部分。

**局限性**：其他智能体的策略也在不断变化，每个智能体面临的环境是非平稳的（non-stationary），可能导致训练不稳定或收敛到次优策略。

**实现位置**：[`agents/iddpg_agent.py`](agents/iddpg_agent.py) + [`experiments/adapters.py`](experiments/adapters.py)（IDDPGManager）

### 5.4 MADDPG（多智能体 DDPG）

**核心思想：集中式训练 + 分布式执行（CTDE）**

- **训练阶段**：集中式 Critic 使用所有智能体的状态和动作信息，从全局角度评估联合动作价值
- **执行阶段**：每个智能体只根据自身局部观测，通过自己的 Actor 做决策

**网络结构**：

```
Actor 网络（每个智能体独立）:
    局部观测 → [FC(64) → ReLU → FC(64) → ReLU → FC(5) → Sigmoid] → 5维连续动作

Critic 网络（集中式）:
    [所有智能体的观测 + 所有智能体的动作] → [FC(128) → ReLU → FC(64) → ReLU → FC(1)] → Q 值
```

**TD3 稳定性增强**：工程上引入三项轻量级改进（不改变 MADDPG 整体框架）：

| 机制 | 做法 | 解决的问题 |
|------|------|-----------|
| 截断双 Q 网络 | 构建两个独立 Critic，取 `min(Q_A, Q_B)` | 压制 Q 值高估 |
| 延迟策略更新 | Critic 每更新 2 次，Actor 才更新 1 次 | 降低策略震荡 |
| 目标策略平滑 | 给 target actor 输出加截断噪声 | 平滑 Q 函数 |

**实现位置**：[`agents/maddpg_agent.py`](agents/maddpg_agent.py)

### 5.5 RS-MADDPG（奖励塑形 MADDPG）

**RS-MADDPG = Reward-Shaped MADDPG**，在 MADDPG 框架基础上针对协同覆盖任务进行奖励函数优化。

#### 改进机制一：目标分配引导奖励

基于贪心最近分配策略，每个时间步为每个智能体分配一个软目标点，引导"一机一目标"的分工模式。

- 使用迟滞机制（hysteresis）防止目标闪烁：只有当备选目标点比当前锁定目标点近超过阈值 ε 时才切换
- 每 episode 前 K 步保持初始分配不变，提供稳定起步信号

**实现位置**：[`rewards/assignment.py`](rewards/assignment.py)

#### 改进机制二：冗余覆盖惩罚

当某目标点的覆盖半径 rc 内同时存在超过 1 个智能体时触发惩罚，处罚重复覆盖行为，促进搜索资源均匀分配。

**实现位置**：[`rewards/redundancy.py`](rewards/redundancy.py)

#### 改进机制三：安全距离约束

设定安全距离阈值 dsafe，当任意两架无人机间距离小于该阈值但尚未碰撞时，给予渐进惩罚（与距离成比例），使智能体提前学会避让。

**实现位置**：[`rewards/safety.py`](rewards/safety.py)

#### 改进机制四：动态奖励权重机制

奖励权重随训练进度分段线性调整：

| 训练阶段 | 覆盖权重 | 安全惩罚 | 冗余惩罚 |
|---------|---------|---------|---------|
| 早期（0–30%） | 1.0 | 低权重 | 低权重 |
| 中期（30–70%） | 1.0 | 中权重 | 中权重 |
| 后期（70–100%） | 1.0 | 满权重 | 满权重 |

**实现位置**：[`rewards/weight_scheduler.py`](rewards/weight_scheduler.py)

#### 综合奖励函数

```
R_i = R_cover,i + α(t)·R_assign,i - β(t)·R_collision,i - γ(t)·R_redundant - δ(t)·R_safe
```

其中 α(t), β(t), γ(t), δ(t) 为随时间动态调整的权重系数。

**实现位置**：[`agents/rs_maddpg_agent.py`](agents/rs_maddpg_agent.py)

---

## 6. 实验设计

### 6.1 实验一：随机策略与训练后策略对比

**目的**：展示智能体学习前后的行为差异。

| 方法 | 预期表现 |
|------|---------|
| Random | 无序移动，覆盖率低，碰撞频繁 |
| MADDPG | 逐渐分散，各覆盖一个目标点，碰撞减少 |
| RS-MADDPG | 进一步减少冗余覆盖，保持安全距离 |

**运行命令**：
```bash
python experiments/exp1_random_vs_trained.py --eval_episodes 100
```

### 6.2 实验二：四算法横向对比（主实验）

**目的**：系统对比四种方法在协同覆盖任务上的性能差异。

| 实验规模 | Episodes | Seeds | 用途 |
|---------|----------|-------|------|
| 快速验证 | 5,000 | 1 | 调参、排查 bug |
| 最终实验 | 20,000 | 3 | 正式结果（均值 ± 标准差） |

**运行命令**：
```bash
# 快速验证
python experiments/exp2_algorithm_comparison.py

# 完整实验
python experiments/exp2_algorithm_comparison.py --full
```

### 6.3 实验三：消融实验

**目的**：验证 RS-MADDPG 中每个改进模块的独立贡献。

| 实验组 | 目标分配 | 冗余惩罚 | 安全距离 | 碰撞惩罚 |
|--------|:-------:|:-------:|:-------:|:-------:|
| MADDPG (baseline) | | | | |
| + Assignment | ✓ | | | |
| + Assignment + Redundancy | ✓ | ✓ | | |
| + Assignment + Redundancy + Safety | ✓ | ✓ | ✓ | |
| RS-MADDPG (full) | ✓ | ✓ | ✓ | ✓ |

**运行命令**：
```bash
python experiments/exp3_ablation.py --episodes 10000 --seeds 42,123
```

### 6.4 实验四：泛化实验

**目的**：测试算法在不同规模任务中的表现。

| 场景 | 智能体数 | 目标点数 | max_cycles |
|------|:------:|:------:|:----------:|
| 基础 | 3 | 3 | 50 |
| 扩展一 | 4 | 4 | 70 |
| 扩展二 | 5 | 5 | 90 |

> **注意**：不同规模分别从头训练（非 Zero-shot），因为 MLP 输入维度随 N 变化。

**运行命令**：
```bash
python experiments/exp4_generalization.py --episodes 10000
```

---

## 7. 评价指标

| 指标 | 定义 | 含义 |
|------|------|------|
| **平均 Episode 总奖励** | 一个 episode 内所有智能体累计奖励的均值 | 衡量整体任务完成效果 |
| **目标覆盖率** | 被至少一个智能体覆盖（距离 ≤ rc）的目标点数 / 总目标点数 | 衡量搜索完成度 |
| **碰撞次数** | 一个 episode 内智能体间碰撞事件总数 | 衡量飞行安全性 |
| **平均最小距离** | 每个目标点到最近智能体距离的均值 | 衡量接近程度 |
| **冗余覆盖率** | 有 ≥ 2 个智能体重复覆盖的目标点数 / 总目标点数 | 衡量搜索资源浪费 |
| **Episode 完成步数** | 所有目标点首次被全部覆盖所需的步数 | 衡量搜索效率 |

指标计算实现在 [`utils/metrics.py`](utils/metrics.py)，训练器每 episode 自动评估并记录。Web 控制台和命令行训练均使用同一套指标计算。

---

## 8. 配置说明

所有超参数集中在 [`configs/default.yaml`](configs/default.yaml) 中管理。

### 8.1 配置文件结构

```yaml
env:                  # 环境配置
  num_agents: 3       # 智能体（目标点）数量
  local_ratio: 0.5    # 局部/全局奖励权重比例
  max_cycles: 50      # 每 episode 最大步数

network:              # 网络超参数
  actor_lr: 0.001     # Actor 学习率
  critic_lr: 0.001    # Critic 学习率
  gamma: 0.95         # 折扣因子
  tau: 0.01           # 软更新系数

training:             # 训练超参数
  buffer_size: 1000000  # 回放池容量
  batch_size: 1024      # 批次大小
  episodes: 20000       # 训练 episode 总数
  eval_interval: 500    # 评估间隔

td3:                  # TD3 稳定性增强
  policy_delay: 2     # Actor 延迟更新步数
  target_noise_std: 0.2
  target_noise_clip: 0.5

rs_maddpg:            # RS-MADDPG 改进模块
  assignment: {lambda_a: 0.5, hysteresis_epsilon: 0.05, lock_steps: 5}
  redundancy: {coverage_radius_ratio: 0.12, lambda_r_max: 0.3}
  safety: {safe_distance_ratio: 0.1, lambda_s_max: 0.5}
  weight_scheduling: {enabled: false}

experiment:           # 实验配置
  seeds: [42, 123, 456]

output:               # 输出目录
  log_dir: "outputs/logs"
  model_dir: "outputs/models"
```

> **注意**：Web 训练控制台（`web_server.py`）使用内置的 `TrainingService.DEFAULT_CONFIG` 作为默认配置，不读取 YAML 文件。命令行实验脚本（`exp2_algorithm_comparison.py` 等）支持自动加载 YAML 配置。

### 8.2 Web 控制台配置

Web 控制台的默认配置位于 [`web_server.py`](web_server.py) 中的 `TrainingService.DEFAULT_CONFIG`：

```python
DEFAULT_CONFIG = {
    "experiment_mode": "single",
    "algo": "RS-MADDPG",
    "episodes": 500,
    "num_agents": 3,
    "actor_lr": 1e-3,
    "critic_lr": 1e-3,
    "gamma": 0.95,
    "batch_size": 256,
    "buffer_warmup": 256,
    "seed": 42,
    "coverage_ratio": 0.12,
    "safe_ratio": 0.10,
    "use_weight_scheduling": True,
    "use_assignment": True,
    "use_redundancy": True,
    "use_safety": True,
    "use_collision": True,
    "use_wandb": False,
    "wandb_project": "uav-marl",
    "wandb_run_name": "",
    "update_repeats": 1,
    "frame_stride": 1,
}
```

这些配置可在浏览器页面左侧修改，并通过 `/api/train/start` 发送到后端。后端会进行范围校验和裁剪，避免非法参数导致训练崩溃。

### 8.3 Web 实验模式

| 模式 | `experiment_mode` | 自动运行内容 | 对应计划书 |
|------|-------------------|--------------|------------|
| 单次训练 | `single` | 当前选择的一个算法 | 调试、快速观察 |
| 实验一：随机 vs 训练后 | `exp1` | Random + 当前训练算法（默认 RS-MADDPG） | 实验一 |
| 实验二：算法对比 | `exp2` | Random / IDDPG / MADDPG / RS-MADDPG | 实验二 |
| 实验三：消融实验 | `exp3` | MADDPG baseline、+Assignment、+Redundancy、+Safety、RS-MADDPG full | 实验三 |
| 实验四：泛化实验 | `exp4` | N=3 / N=4 / N=5 的 RS-MADDPG | 实验四 |

> 批量实验会按 run 队列依次执行。每个 run 结束后保存日志；整个实验完成后自动生成结构化总结。

### 8.4 可选服务配置

| 功能 | 配置方式 | 说明 |
|------|----------|------|
| W&B | 页面勾选「启用 W&B」并填写 Project / Run Name | 记录长训练曲线与模型文件 |
| DeepSeek | 启动前设置 `DEEPSEEK_API_KEY` 环境变量 | 根据结构化指标生成自然语言报告 |
| 端口 | `python web_server.py --port 8600` | 默认 `8600` |
| Host | `python web_server.py --host 127.0.0.1` | 默认只监听本机 |

### 8.5 关键超参数调参建议

| 参数 | 建议范围 | 说明 |
|------|---------|------|
| `actor_lr` / `critic_lr` | 1e-4 ~ 1e-2 | 学习率过高导致震荡，过低收敛慢 |
| `gamma` | 0.90 ~ 0.99 | 折扣因子，任务步数少时适当降低 |
| `tau` | 0.005 ~ 0.05 | 软更新系数，越小目标网络更新越慢 |
| `batch_size` | 512 ~ 2048 | 批次大小，受内存限制 |
| `buffer_size` | 1e5 ~ 1e6 | 回放池容量，至少为 batch_size 的 100 倍 |

---

## 9. Web 训练控制台

项目提供一套纯 HTML/CSS/JS + Python 标准库的工业风训练控制台，无需 Streamlit、FastAPI、Node.js 等任何外部框架。

### 9.1 架构设计

```
┌─────────────────────────────────────────────────────────┐
│  Browser (web_static/)                                   │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ index.html│  │   app.js     │  │   styles.css     │  │
│  │ DOM 结构  │  │ Canvas 2D    │  │   暗色主题        │  │
│  │           │  │ SSE 事件流    │  │   响应式布局      │  │
│  │           │  │ REST API 调用 │  │                   │  │
│  └──────────┘  └──────────────┘  └──────────────────┘  │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP + SSE
┌───────────────────────▼─────────────────────────────────┐
│  Python Server (web_server.py)                           │
│  ┌──────────────────┐  ┌──────────────────────────────┐ │
│  │ RequestHandler   │  │  TrainingService             │ │
│  │ GET/POST /api/*  │  │  后台训练线程                 │ │
│  │ SSE /api/events  │  │  EventHub 事件广播            │ │
│  │ 静态文件服务      │  │  自动保存日志/模型            │ │
│  └──────────────────┘  └──────────────────────────────┘ │
│                                                          │
│  IndustrialHTTPServer (ThreadingHTTPServer)               │
└─────────────────────────────────────────────────────────┘
```

### 9.2 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 服务 index.html |
| `/styles.css` | GET | 样式表 |
| `/app.js` | GET | 前端 JS |
| `/api/state` | GET | 获取当前训练状态快照 |
| `/api/events` | GET | SSE 事件流（实时推送训练数据） |
| `/api/train/start` | POST | 启动训练（传入配置 JSON） |
| `/api/train/stop` | POST | 停止训练 |
| `/api/report/deepseek` | POST | 调用 DeepSeek 基于结构化总结生成自然语言报告 |

### 9.3 SSE 事件类型

| 事件 | 触发时机 | 携带数据 |
|------|---------|---------|
| `status` | 状态变更 | 完整训练状态快照 |
| `run_start` | 批量实验中某个 run 开始 | run 序号、算法、标签 |
| `run_end` | 某个 run 结束 | run 结果、历史日志 |
| `episode_start` | 每 episode 开始 | 地标位置、覆盖/安全半径 |
| `frame` | 每环境步（按 frame_stride 采样） | 智能体位置、episode/step |
| `episode_end` | 每 episode 结束 | 评估指标、进度百分比 |
| `report` | 整个实验完成 | 结构化实验总结 |
| `llm_report` | DeepSeek 返回 | 自然语言报告文本 |
| `complete` | 训练完成 | 最终状态快照 |
| `error` | 训练异常 | 错误信息 |
| `wandb` | W&B 连接状态变化 | 项目名、运行名、URL |

### 9.4 自动结果总结

Web 控制台在实验结束后会调用 [`reports/report_generator.py`](reports/report_generator.py)，根据每个 run 的 episode 日志生成结构化总结：

- **核心指标**：平均奖励、目标覆盖率、碰撞次数、平均最小距离、冗余覆盖率、完成步数
- **稳健统计**：最后 10% episode 的均值、标准差、最大值、最小值
- **趋势判断**：后半段奖励是否提升、覆盖率是否提升、覆盖率是否稳定
- **综合评分**：覆盖率、安全性、冗余控制、平均最小距离、完成效率的加权评分（0-100）
- **横向对比**：多 run 实验中自动排序，找出当前最优算法/设置
- **诊断建议**：指出覆盖不足、碰撞偏多、冗余偏高、训练轮数不足等问题
- **报告文本**：生成一段不依赖 LLM 的基础报告文字，可直接作为课程报告草稿

结构化总结会保存到：

```bash
outputs/logs/<experiment_mode>_seed<seed>_summary.json
```

每个 run 的原始日志会保存到：

```bash
outputs/logs/<run_id>_seed<seed>_logs.json
```

### 9.5 DeepSeek 报告生成

DeepSeek 调用流程：

1. 后端先生成结构化总结 JSON
2. 用户点击页面中的「DeepSeek 生成报告」
3. 后端读取 `DEEPSEEK_API_KEY`
4. 调用 `https://api.deepseek.com/chat/completions`
5. 将自然语言报告返回前端并显示在「报告文本」区域

设计原则：LLM 只基于结构化指标写结论，不直接接触训练过程，也不允许编造未出现的数据。

### 9.6 配色体系

```
智能体：#55a7ff（天蓝） / #ff6b6b（珊瑚红） / #f4c430（琥珀黄）
       #59d98e（薄荷绿） / #c77dff（淡紫）

目标点：#f4c430（金色菱形）   覆盖圈：rgba(89,217,142,0.55)（绿色虚线）
安全圈：半透明智能体色   轨迹拖尾：同色渐隐

UI：背景 #0b0f14   面板 #111821   强调色 #2ec4b6   文字 #eef4f8
```

### 9.7 启动命令

```bash
python web_server.py --port 8600
# 可选参数：--host 127.0.0.1（默认）、--port 8600（默认）
```

浏览器打开 `http://127.0.0.1:8600`，配置参数后点击「开始训练」。

### 9.8 与 Streamlit 旧版的关系

`app.py` 和 `pages/` 目录为早期基于 Streamlit 的前端实现，保留作为备选方案。主要差异：

| 特性 | Web 控制台 (web_server.py) | Streamlit (app.py) |
|------|---------------------------|-------------------|
| 前端框架 | 纯 HTML/CSS/JS | Streamlit (Python) |
| 服务器 | Python stdlib | Streamlit 内置 |
| 实时渲染 | Canvas + SSE 推送 | image/pyplot 轮询 |
| 外部依赖 | 无 | streamlit, plotly |
| 训练架构 | 后台线程 | 后台线程 |
| 实验页面 | 事件流实时 | 独立 .py 页面 |

---

## 10. 测试

### 10.1 测试覆盖

| 测试文件 | 覆盖模块 | 测试数 |
|---------|---------|:-----:|
| `test_agents.py` | 智能体基类、各算法智能体 | 多 |
| `test_env_wrapper.py` | 环境封装、状态空间、动作空间 | 多 |
| `test_networks.py` | Actor / Critic 网络前向传播 | 多 |
| `test_rewards.py` | 目标分配、冗余惩罚、安全距离、权重调度 | 25 |
| `test_rs_maddpg.py` | RS-MADDPG 集成测试（含 save/load） | 10 |
| `test_utils.py` | 经验回放池、OU 噪声、评价指标 | 多 |

### 10.2 运行测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行指定模块
pytest tests/test_rewards.py -v
pytest tests/test_rs_maddpg.py -v

# 带覆盖率报告
pytest tests/ --cov=. --cov-report=term-missing
```

---

## 11. GitHub 上传前检查

本项目可以上传到 GitHub，但不要直接把整个文件夹拖进网页上传。上传前请确认以下内容：

### 11.1 不能上传的本地文件

以下文件和目录已经由 `.gitignore` 忽略：

| 路径 | 原因 |
|------|------|
| `.env.local` | 保存本机 DeepSeek API Key，属于私密配置 |
| `.env` / `.env.*` | 可能包含环境变量或密钥 |
| `wandb/` | W&B 本地运行日志，文件大且可重新生成 |
| `outputs/**` | 训练日志、模型权重、导出图片等运行产物 |
| `.venv/` / `venv/` | Python 虚拟环境 |
| `.pytest_cache/` / `__pycache__/` | 测试缓存和 Python 缓存 |

`outputs/`、`outputs/logs/`、`outputs/models/`、`outputs/exports/` 会通过 `.gitkeep` 保留空目录结构，但不会提交具体训练结果。

### 11.2 密钥配置方式

仓库只提交 `.env.example`，使用者可以复制成 `.env.local` 后填入自己的 DeepSeek API Key：

```bash
cp .env.example .env.local
```

也可以直接双击 `start_uav_marl.command`，脚本会在首次启动时提示输入密钥并自动写入 `.env.local`。

### 11.3 推荐上传流程

```bash
git init
git status --ignored
git add .
git status --short
git commit -m "Initial UAV MARL project"
```

提交前重点检查 `git status --short`：不应该出现 `.env.local`、`wandb/`、`.venv/`、`.pytest_cache/`，也不应该出现 `outputs/logs/*.json` 或 `outputs/models/*.pt`。

如果使用 GitHub 网页拖拽上传，请手动排除 `.env.local`、`wandb/`、`.pytest_cache/`、`.venv/`、`outputs/logs/`、`outputs/models/` 中的运行产物。

---

## 12. 参考文献

本项目参考以下 10 篇核心文献，按三阶段实施计划分组：

### 第一阶段：环境与任务建模

| 编号 | 文献 |
|:----:|------|
| [1] | J. K. Terry et al. "PettingZoo: Gym for Multi-Agent Reinforcement Learning." *NeurIPS*, 2021. |
| [2] | R. Lowe, Y. Wu, A. Tamar et al. "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments." *NeurIPS*, 2017. |
| [3] | L. Busoniu, R. Babuska, and B. De Schutter. "A Comprehensive Survey of Multiagent Reinforcement Learning." *IEEE TSMCC*, 2008. |

### 第二阶段：基础算法

| 编号 | 文献 |
|:----:|------|
| [4] | T. P. Lillicrap et al. "Continuous Control with Deep Reinforcement Learning." *ICLR*, 2016. |
| [5] | M. Tan. "Multi-Agent Reinforcement Learning: Independent vs. Cooperative Agents." *ICML*, 1993. |
| [6] | J. Foerster et al. "Stabilising Experience Replay for Deep Multi-Agent Reinforcement Learning." *ICML*, 2017. |

### 第三阶段：改进机制

| 编号 | 文献 |
|:----:|------|
| [7] | A. Y. Ng, D. Harada, and S. Russell. "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping." *ICML*, 1999. |
| [8] | J. Cortes et al. "Coverage Control for Mobile Sensing Networks." *IEEE TAC*, 2004. |
| [9] | A. Khamis, A. Hussein, and A. Elmogy. "Multi-robot Task Allocation: A Review of the State-of-the-Art." Springer, 2015. |
| [10] | J. García and F. Fernández. "A Comprehensive Survey on Safe Reinforcement Learning." *JMLR*, 2015. |

---

## 13. 项目信息

### 13.1 项目背景

本项目为课程设计项目，目标是在三周周期内完成从环境搭建、算法实现、实验对比到可视化汇报的完整流程。项目围绕无人机协同区域搜索问题，基于 PettingZoo MPE 环境构建多智能体协同覆盖任务，并采用多智能体强化学习方法训练无人机集群实现目标覆盖和碰撞规避。

### 13.2 预期成果

1. 可运行的多智能体无人机协同覆盖仿真程序
2. 一套基于 PettingZoo MPE 的实验环境配置
3. Random / IDDPG / MADDPG / RS-MADDPG 四种方法的对比实验数据
4. 覆盖覆盖率、碰撞次数、冗余率、完成步数等多维度的训练曲线
5. 工业风 Web 训练控制台（HTML Canvas 实时 2D 渲染 + SSE 事件流）
6. 消融实验分析（验证每个改进模块的独立贡献）
7. 不同规模下的泛化实验结果
8. 自动结果总结（结构化指标、诊断建议、报告文本）
9. DeepSeek 辅助生成自然语言实验结论
10. 完整的项目报告和汇报材料

### 13.3 作者

<!-- 请在此处填写作者信息 -->

### 13.4 许可证

本项目采用 [MIT License](LICENSE) 开源。

---

> **项目详细设计文档**：参见 [`项目计划书.md`](../项目计划书.md)
