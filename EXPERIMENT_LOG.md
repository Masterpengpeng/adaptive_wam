# Adaptive Inference for WAM — 实验记录与 Codex 协作指南

> 文件定位：本文件是本课题的长期实验主记录（Source of Truth）。
>
> 用途：记录实验目标、环境、代码修改、参数、结果、失败原因、阶段结论与下一步计划，并作为与 Codex 沟通时的统一上下文。
>
> 原则：**先复现，再修改；一次只改一个核心变量；所有结论必须能追溯到具体实验。**

---

## 0. 项目基本信息

### 研究题目
**Adaptive Inference for World Action Models**

### 当前核心思想

让 World Action Model 根据当前状态，自适应调整推理配置：

- **H — Horizon**：未来预测多远
- **D — Depth**：未来生成 / 去噪多深
- **W — Width**：考虑多少个未来 rollout
- **E — Execution Horizon**：一次规划后连续执行多少步再重新观察

核心目标不是单纯减少计算，而是改善：

**Task Performance ↔ Latency ↔ Compute**

之间的权衡。

### 当前核心假设

不同机器人状态对应的最优推理配置不同：

```text
简单状态：
H ↓, D ↓, W ↓, E ↑
=> 少想，多执行

困难 / 高风险状态：
H ↑, D ↑, W ↑, E ↓
=> 多想，少执行
```

当前最重要的前置问题：

> 不同 state 的最优 H / D / W / E 是否真的具有明显差异？

如果没有明显 state dependence，则 Adaptive Inference 的研究价值会大幅下降。

---

# 1. 服务器项目目录

计划在服务器用户目录下创建：

```bash
mkdir -p ~/ZP_projects
cd ~/ZP_projects
```

建议为当前课题单独建立目录：

```bash
mkdir -p ~/ZP_projects/adaptive_wam
cd ~/ZP_projects/adaptive_wam
```

推荐目录结构：

```text
~/ZP_projects/adaptive_wam/
│
├── repos/                  # 第三方 baseline 仓库
│   └── FastWAM/
│
├── configs/                # 自己维护的实验配置
│   ├── baseline/
│   ├── sweep_E/
│   ├── sweep_D/
│   ├── sweep_H/
│   └── sweep_W/
│
├── scripts/                # 自己写的实验脚本
│   ├── eval/
│   ├── profiling/
│   └── oracle/
│
├── checkpoints/            # checkpoint（不要提交到 git）
│
├── data/                   # 数据 / benchmark 资源（不要提交大文件）
│
├── outputs/                # 原始实验输出
│   ├── baseline/
│   ├── sweep_E/
│   ├── sweep_D/
│   ├── sweep_H/
│   ├── sweep_W/
│   └── oracle/
│
├── results/                # 整理后的 csv / json / 图表
│
├── logs/                   # stdout / stderr / runtime logs
│
├── notes/                  # 论文阅读、代码理解、临时记录
│
├── EXPERIMENT_LOG.md       # 本文件：实验主记录
│
└── README.md               # 项目入口说明
```

### 目录原则

1. 第三方仓库放在 `repos/`，不要直接在根目录乱改。
2. 原始实验输出和整理后的结果分开。
3. 不覆盖旧结果。
4. 每一组实验使用独立目录。
5. checkpoint、dataset、日志、大结果文件不要随意提交 Git。
6. 所有重要修改必须留下 Git commit。
7. 服务器登录密码等敏感信息**不要写入本文件或 Git 仓库**。

---

# 2. 当前硬件条件

服务器 GPU：

```text
10 × NVIDIA RTX 2080 Ti
```

实验策略：

- 初期 Debug：优先只使用 1 张 GPU。
- 单任务跑通后：使用 2 张 GPU 做复现验证。
- 批量 evaluation / sweep 稳定后：再考虑多 GPU 并行。
- 不要一开始占满 10 张 GPU。
- 2080 Ti 单卡显存有限，应优先进行 checkpoint inference、evaluation、Oracle Study 和轻量 Controller 训练。
- 不默认从头训练大型 WAM。

### Latency 记录原则

所有方法必须：

- 同一台服务器
- 同一种 GPU
- 相同 batch setting
- 相同 benchmark
- 相同计时方法

优先报告：

```text
Relative Latency
Relative Compute
Relative WAM Calls
```

而不是直接和其他论文不同 GPU 上的绝对毫秒数比较。

---

# 3. 给 Codex 的第一条总控指令

进入项目目录以后，把下面内容作为 Codex 的项目级说明。

```text
你是本项目的科研工程执行助手。

项目名称：
Adaptive Inference for World Action Models

项目目标：
研究 World Action Model 在 inference 阶段是否应该根据当前 state
动态调整 H / D / W / E，从而在保持任务性能的情况下减少不必要的推理计算，
或在相同计算预算下提高任务成功率。

四个核心变量：

H = future prediction / imagination horizon
D = denoising / refinement depth
W = number of stochastic future rollouts
E = execution horizon / action chunk commitment length

当前阶段最重要的任务不是直接设计 Adaptive Controller，
而是首先完成 baseline reproduction 和 single-axis sensitivity study，
验证不同 state / task phase 的最优 H / D / W / E 是否不同。

当前计划：

1. 创建并检查项目目录
2. 获取并阅读 baseline 代码
3. 配置环境
4. 复现官方 checkpoint evaluation
5. 找出代码中 E / D / H / W 分别对应的位置
6. 按顺序进行 E → D → H → W 单轴实验
7. 进行 state-level counterfactual Oracle Study
8. Oracle 现象成立后，才开始训练 Adaptive Controller

工作规则：

1. 在修改任何代码前，先阅读相关文件并解释现有逻辑。
2. 不要一次同时修改 H / D / W / E。
3. 每次修改前说明：
   - 修改目的
   - 涉及文件
   - 预计影响
4. 修改后必须给出 diff / 修改摘要。
5. 不覆盖旧实验输出。
6. 不删除第三方代码、checkpoint 或结果，除非我明确允许。
7. 运行耗时较长的实验前，先告诉我预计使用几张 GPU、预计命令和输出目录。
8. 遇到报错时，先定位原因，不要连续进行大范围无关修改。
9. 每完成一个实验阶段，都输出 Experiment Handoff。
10. Experiment Handoff 必须包含：
    - 实验目标
    - Git commit
    - 环境
    - GPU
    - 修改文件
    - 实验命令
    - 参数
    - 输出目录
    - 核心结果
    - 异常
    - 当前结论
    - 下一步建议

不要因为追求“尽快跑起来”而跳过实验记录。
本项目优先保证可复现性、可追踪性和单变量实验逻辑。
```

---

# 4. 第一阶段：项目初始化

## 目标

只完成：

```text
目录建立
→ Git 初始化
→ baseline repo 准备
→ 环境信息记录
```

暂时不要改算法。

## 推荐命令

```bash
cd ~/ZP_projects
mkdir -p adaptive_wam
cd adaptive_wam

mkdir -p repos configs scripts checkpoints data outputs results logs notes
```

如果当前项目还没有 Git：

```bash
git init
```

建议创建 `.gitignore`：

```gitignore
# checkpoints / data
checkpoints/
data/

# outputs
outputs/
logs/

# Python
__pycache__/
*.pyc
.venv/
venv/

# IDE
.vscode/
.idea/

# misc
*.log
```

## 让 Codex 做的第一件事

```text
现在先不要修改任何算法代码。

请完成以下任务：

1. 检查当前目录结构。
2. 如果缺少必要目录，给出建议，但不要删除已有目录。
3. 检查：
   - nvidia-smi
   - Python
   - CUDA
   - gcc
   - conda
   - git
4. 把环境信息整理成一个简洁报告。
5. 暂时不要安装大型依赖。
6. 告诉我下一步适合开始 baseline clone，还是当前环境还有明显问题。
```

### 环境记录

```text
日期：
Hostname：
GPU：
Driver：
CUDA：
Python：
Conda：
PyTorch：
Git：
备注：
```

---

# 5. 第二阶段：Baseline 获取与代码阅读

## 当前候选 Baseline

优先：

```text
Fast-WAM / JointWAM
Benchmark: LIBERO
```

原则：

> 先确认该 baseline 当前公开代码、checkpoint、evaluation 流程和 H/D/W/E 可修改性，再开始实验。

## 给 Codex 的指令

```text
请先阅读 baseline 项目的 README、evaluation 脚本和主要 inference 路径。

这一轮不要修改代码，也不要开始长时间训练。

请回答：

1. 官方推荐的环境是什么？
2. checkpoint 如何获取？
3. LIBERO evaluation 入口在哪里？
4. 一次完整 inference 的调用链是什么？
5. action 是在哪里生成的？
6. action chunk 在哪里执行？
7. future / imagination 在哪里生成？
8. diffusion / denoising loop 在哪里？
9. 是否原生支持多 future samples？
10. 哪些代码位置最可能对应 H / D / W / E？

最后输出：
- 关键文件路径
- 关键函数
- 调用链
- H/D/W/E 的初步映射表
- 仍不确定的问题

不要修改代码。
```

## 代码理解记录

```text
Baseline：
Repo：
Branch：
Commit：

Inference Entry：
Evaluation Entry：

H 可能对应：
D 可能对应：
W 可能对应：
E 可能对应：

尚未确认：
```

---

# 6. 第三阶段：官方 Baseline 复现

## 目标

做到：

```text
环境可运行
checkpoint 可加载
LIBERO task 可启动
至少一个 task 完整 evaluation 成功
结果可保存
latency 可测量
```

这一阶段：

**不要修改 H / D / W / E。**

## 给 Codex 的指令

```text
现在只做官方 baseline reproduction。

要求：

1. 优先使用官方 checkpoint。
2. 不修改模型算法。
3. 先选择最小规模的 evaluation 配置验证运行链路。
4. 初次只使用 1 张 GPU。
5. 成功后再扩大任务数。
6. 保存完整运行命令。
7. 保存 stdout/stderr。
8. 记录：
   - success rate
   - inference latency
   - GPU memory
   - episode 数量
   - 随机种子
9. 结果保存在 outputs/baseline/ 下。
10. 如果结果与官方差异较大，不要进入下一阶段，先定位原因。

结束后输出 Experiment Handoff。
```

## Baseline 复现实验记录模板

### EXP-B000 — Baseline Reproduction

```text
日期：
负责人：
Baseline：
Benchmark：
Git Commit：

GPU：
CUDA：
PyTorch：

Checkpoint：

Command：

Config：

Seed：
Episodes：
Tasks：

Success Rate：
Mean Latency：
Peak GPU Memory：

Output：
Log：

是否与官方结果一致：
差异：

问题：

结论：

下一步：
```

---

# 7. 第四阶段：Single-Axis Sensitivity Study

重要原则：

> **一次只改变一个变量，其余全部固定。**

推荐顺序：

```text
E → D → H → W
```

---

## 7.1 E Sweep

### 研究问题

不同 execution horizon 是否产生明显 performance / compute 差异？

候选值：

```text
E ∈ {1, 2, 4, 8}
```

H / D / W 固定为 baseline 默认值。

### 给 Codex 的指令

```text
现在开始 E-only sensitivity study。

要求：

1. 先找出 execution horizon / action chunk commitment 的实际控制位置。
2. 修改前解释现有逻辑。
3. 将 E 做成 config 参数，不要 hard-code。
4. 支持至少：
   E = 1, 2, 4, 8
5. H / D / W 保持完全不变。
6. 相同 tasks、相同 seeds、相同 checkpoint。
7. 每个 E 独立输出目录。
8. 记录：
   - success rate
   - latency
   - WAM calls per episode
   - episode length
   - GPU memory
9. 不覆盖 baseline。
10. 修改完成后先用最小任务测试，确认正确再批量运行。

输出 Experiment Handoff。
```

### E Sweep 结果

| E | Success | Mean Latency | WAM Calls | Episode Length | Notes |
|---|---:|---:|---:|---:|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 8 |  |  |  |  |  |

### 当前结论

```text
待实验。
```

---

## 7.2 D Sweep

候选值：

```text
D ∈ {2, 4, 8, 16}
```

### 重点

确认 D 真正表示的是：

```text
video/world generation refinement
or
denoising inference steps
```

而不是随意选择一个“网络层数”。

### Codex 指令

```text
开始 D-only sensitivity study。

首先确认 baseline 中控制 inference-time denoising / refinement depth 的真实参数。

不要根据变量名猜测。

要求：
1. 阅读 sampler / scheduler / denoising loop。
2. 给出 D 与实际代码的映射。
3. 将 D 暴露为 config。
4. H / W / E 固定。
5. 测试 D = 2, 4, 8, 16（如 baseline 机制不支持这些值，先报告原因）。
6. 记录 success、latency、GPU memory。
7. 每组结果独立保存。
8. 先小规模 sanity check，再批量实验。
```

### D Sweep 结果

| D | Success | Mean Latency | GPU Memory | Notes |
|---|---:|---:|---:|---|
| 2 |  |  |  |  |
| 4 |  |  |  |  |
| 8 |  |  |  |  |
| 16 |  |  |  |  |

---

## 7.3 H Sweep

候选值：

```text
H ∈ {0, 2, 4, 8}
```

特殊关注：

```text
H = 0
```

表示不进行显式 test-time imagination / future rollout。

### Codex 指令

```text
开始 H-only sensitivity study。

首先确认 future prediction horizon 在 baseline 中的真实含义和控制位置。

重点检查是否存在：
- future frame number
- rollout horizon
- imagination horizon
- no-imagination / action-only branch

H = 0 必须有清晰定义。
如果 baseline 原生不支持 H = 0，不要偷偷使用近似替代，
先解释需要怎样实现 action-only / no-imagination branch。

其余 H 值实验时 D / W / E 保持固定。
记录 success、latency、compute、WAM calls。
```

### H Sweep 结果

| H | Success | Mean Latency | Compute | Notes |
|---|---:|---:|---:|---|
| 0 |  |  |  |  |
| 2 |  |  |  |  |
| 4 |  |  |  |  |
| 8 |  |  |  |  |

---

## 7.4 W Sweep

候选值：

```text
W ∈ {1, 2, 4}
```

W 表示同一 state 下的多个 stochastic future rollouts。

2080 Ti 显存有限。

优先考虑：

```text
sequential rollout
```

而不是简单把 W 全部堆进单卡 batch。

### Codex 指令

```text
开始 W-only sensitivity study。

首先确认 baseline 是否原生支持多个 stochastic future samples。

如果不支持，请先设计最小侵入式实现方案。

要求：

1. W = 1, 2, 4。
2. H / D / E 固定。
3. 保持每个 future sample 的采样逻辑一致。
4. 避免因 W 增大导致单卡 OOM：
   优先 sequential rollout 或合理多 GPU 分发。
5. 记录总 latency 和每个 rollout latency。
6. 记录不同 rollout 之间的 disagreement。
7. 暂时不要设计复杂融合器。
8. 先回答：
   “增加 W 是否真的提高任务表现？”
```

### W Sweep 结果

| W | Success | Mean Latency | Rollout Disagreement | Notes |
|---|---:|---:|---:|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 4 |  |  |  |  |

---

# 8. 第五阶段：State-Level Oracle Study

这是本课题当前最重要的 **Go / No-Go 实验**。

## 核心问题

对于同一个状态：

```text
same state
+
different inference configurations
```

最优 H / D / W / E 是否不同？

### 理想实验逻辑

```text
保存 state s_t
        │
        ├── B1 = (H1,D1,W1,E1)
        ├── B2 = (H2,D2,W2,E2)
        ├── B3 = (H3,D3,W3,E3)
        └── ...
              ↓
       从相同 state 继续完成 episode
              ↓
         估计 Q(s_t, B)
```

目标：

```text
B*_t = argmax_B [Q(s_t,B) - λ C(B)]
```

### 对比

```text
Global Fixed Oracle
Task-wise Oracle
State-wise Oracle
```

最关键结果：

```text
State-wise Oracle >> Global Fixed Oracle
```

如果这个 gap 很小：

> 暂缓训练 Adaptive Controller，重新审视研究假设。

如果 gap 明显：

> 进入 Adaptive Controller 阶段。

### Codex 指令

```text
现在设计 Counterfactual State-Level Oracle Study。

这一轮先不要直接写大量代码。

请先分析 simulator / benchmark 是否支持：
1. 保存环境 state
2. 恢复完全相同 state
3. 控制随机数状态
4. 从同一个 state 分叉执行多个 inference configuration
5. 独立完成后续 episode

请输出：
- 可行性
- 涉及 API
- 需要修改的文件
- state serialization 方案
- seed / stochasticity 控制方案
- Q(s,B) 的候选定义
- compute cost C(B) 的记录方案
- 预计实验规模

确认方案后再实现。
```

---

# 9. Adaptive Controller 阶段（暂不执行）

只有 Oracle Study 明确支持 state-dependent configuration 后再做。

初步方案：

```text
Stage 1:
state / WAM hidden representation
        ↓
Imagination Budgeter
        ↓
(H, D, W)

Stage 2:
actual imagined futures
        ↓
rollout disagreement
action disagreement
denoising convergence
world-action consistency
        ↓
Commitment Controller
        ↓
E
```

暂不优先使用复杂 RL。

第一版优先：

```text
Oracle labels
→ Supervised / Imitation Learning Controller
```

之后再考虑：

```text
budget-aware controller
contextual bandit
RL
```

---

# 10. 每次给 Codex 的固定实验指令模板

以后不要只对 Codex 说：

```text
“帮我跑一下实验。”
```

使用下面模板。

```text
【实验编号】
EXP-XXXX

【实验目的】
本实验想回答什么问题？

【研究假设】
如果假设成立，预期看到什么？

【本次允许修改】
只允许修改哪些变量 / 文件？

【本次禁止修改】
哪些内容必须保持固定？

【实验配置】
Baseline：
Checkpoint：
Benchmark：
Tasks：
Seeds：
H：
D：
W：
E：

【硬件限制】
允许使用 GPU：
最大 GPU 数量：
是否允许长时间运行：

【输出要求】
结果保存到：
日志保存到：

必须记录：
- command
- git commit
- config
- success rate
- latency
- GPU memory
- WAM calls
- abnormal episodes

【结束条件】
什么时候算实验完成？

【实验结束后】
请输出 Experiment Handoff，
不要只回复“运行成功”。
```

---

# 11. Experiment Handoff 固定模板

每次 Codex 完成一轮工作，要求复制下面内容并填写。

```text
# Experiment Handoff

## Experiment ID

## Goal

## Hypothesis

## Date

## Git
Branch:
Commit:

## Environment
GPU:
Driver:
CUDA:
Python:
PyTorch:
Conda Env:

## Baseline
Repo:
Checkpoint:
Benchmark:
Tasks:
Seeds:

## Variables
H:
D:
W:
E:

## Changed Files
1.
2.
3.

## Change Summary

## Commands

## Output Paths

## Results
Success Rate:
Mean Latency:
P50 Latency:
P95 Latency:
Peak GPU Memory:
WAM Calls:
Episode Count:

## Errors / Abnormal Behavior

## Interpretation

## Does the result support the hypothesis?

## Open Questions

## Recommended Next Step
```

---

# 12. 实验主记录

> 每完成一个正式实验，在这里追加，而不是修改掉过去的记录。

---

## EXP-000 — 项目初始化

**状态：已完成（baseline 专用 Python 环境待创建）**

### Goal

```text
创建项目目录，确认服务器环境，建立 Git 和实验记录体系。
```

### Result

```text
日期：2026-09-13
负责人：Codex
Git Commit：7594e3f2797c92358e60fdb935f133397a7aead0

项目目录：/home/lab416/ZP_projects/adaptive_wam
目录结构：已创建
Git：已初始化，默认分支 main

Hostname：lab416-server
GPU：10 × NVIDIA GeForce RTX 2080 Ti，单卡 11264 MiB
Driver：570.211.01
CUDA Toolkit：12.8（nvcc V12.8.61）
Python：3.12.7（Conda base）
Conda：24.9.2
GCC：11.4.0
Git：2.34.1

PyTorch：base 中用户级 torch 2.13.0 导入失败，安装不完整。
磁盘：工作区所在文件系统剩余约 55 GiB。
详细记录：notes/environment_2026-09-13.md
```

### Conclusion

```text
GPU、驱动、CUDA 编译器和系统工具链可用，可以进入 baseline 获取与代码阅读阶段。
不得直接使用当前 base Python 环境；应根据选定 baseline 的官方依赖创建独立 Conda 环境，
并在 evaluation 前验证 torch import、CUDA 可见性和最小 tensor 运算。
```

### Next

```text
确认 Fast-WAM / JointWAM 的官方仓库、commit、checkpoint 与 LIBERO evaluation 入口；
随后创建隔离环境。暂不修改算法。
```

---

## EXP-B000 — Baseline Reproduction

**状态：未开始**

### Goal

```text
复现官方 checkpoint evaluation。
```

### Result

```text
待填写。
```

### Conclusion

```text
待填写。
```

---

## EXP-E001 — E Sweep

**状态：未开始**

### Goal

```text
测试 E ∈ {1,2,4,8}。
```

### Result

```text
待填写。
```

---

## EXP-D001 — D Sweep

**状态：未开始**

---

## EXP-H001 — H Sweep

**状态：未开始**

---

## EXP-W001 — W Sweep

**状态：未开始**

---

## EXP-O001 — State-Level Oracle

**状态：未开始**

---

# 13. 失败实验记录

失败实验不能删除。

模板：

```text
## FAILED-XXX

日期：

原实验目标：

使用配置：

出现问题：

错误日志：

根因：

尝试过的修复：

最终状态：
[ ] 已解决
[ ] 暂时搁置
[ ] 方案废弃

为什么：

后续是否还需要重新尝试：
```

---

# 14. Decision Log

用于记录关键科研决策，避免以后忘记“为什么当时这样做”。

---

## DEC-001 — 先做 Oracle，再做 Controller

**Decision**

```text
当前阶段不直接训练 Adaptive Controller。
```

**Reason**

```text
必须先验证 state-dependent optimal H/D/W/E 是否真实存在。
否则 Controller 缺乏研究动机。
```

---

## DEC-002 — 一次只改变一个轴

**Decision**

```text
Single-axis sensitivity 按 E → D → H → W 进行。
```

**Reason**

```text
便于定位每个 inference knob 对 performance / compute 的独立影响，
也降低 Debug 难度。
```

---

## DEC-003 — 不优先从头训练大型 WAM

**Decision**

```text
前期优先使用公开 checkpoint 做 inference / evaluation。
```

**Reason**

```text
当前服务器为 10×2080Ti，适合推理、批量 evaluation、
Oracle Study 和轻量 Controller，但不适合照搬大型 WAM 的高显存训练配置。
```

---

# 15. 当前 To-Do List

## P0 — 立即执行

- [x] 创建 `~/ZP_projects/adaptive_wam`
- [x] 创建标准目录
- [x] 初始化 Git
- [x] 记录服务器环境
- [x] 获取 baseline repo
- [x] 阅读 README 和 evaluation 入口
- [x] 确认 checkpoint 获取方式
- [x] 找出 H / D / W / E 的实际代码映射

## P1 — Baseline

- [ ] 单 GPU 跑通一个最小 evaluation
- [ ] 保存 baseline logs
- [ ] 测量 latency
- [ ] 验证成功率
- [ ] 固定 baseline commit

## P2 — Sensitivity

- [ ] E sweep
- [ ] D sweep
- [ ] H sweep
- [ ] W sweep

## P3 — Core Hypothesis

- [ ] state save / restore
- [ ] counterfactual branches
- [ ] Global Fixed Oracle
- [ ] Task-wise Oracle
- [ ] State-wise Oracle

## P4 — Method

- [ ] 根据 Oracle 结果决定是否继续
- [ ] 设计 Imagination Budgeter
- [ ] 设计 Commitment Controller
- [ ] 训练与 baseline 对比

---

# 16. 当前阶段绝对不要做的事情

- 不要一开始同时修改 H / D / W / E。
- 不要没有 baseline reproduction 就报告“提升”。
- 不要只看单个 episode 得出结论。
- 不要覆盖旧 checkpoint / output。
- 不要在没有 Git commit 的情况下进行重要修改。
- 不要只记录成功实验。
- 不要把服务器密码、Token 等写进本文件。
- 不要因为一次 OOM 就立即大改模型结构。
- 不要用不同硬件直接比较 absolute latency。
- 不要在 Oracle 假设尚未验证时投入大量时间训练复杂 Controller。

---

# 17. 每次结束科研工作前必须更新

每次当天结束前，至少补全：

```text
Today:
今天完成了什么？

Experiments:
今天跑了哪些实验？

Key Results:
最重要结果是什么？

Problems:
还有哪些报错 / 疑问？

Decisions:
今天做了什么重要决定？

Git:
最新 commit 是什么？

Next:
下一次打开服务器后第一件事做什么？
```

---

# 18. 当前一句话状态

```text
当前项目处于“实验准备阶段”。

目标不是马上训练 Adaptive Controller，
而是先搭建 Fast-WAM / WAM baseline 的可复现实验链路，
随后完成 E → D → H → W 单轴 sensitivity study，
最后通过 State-Level Oracle Study 验证 Adaptive Inference
是否具有真实的 state dependence。
```

---

# 19. Daily Log

## 2026-09-13

### Today

完成 `EXP-000`：创建标准目录、初始化独立 Git 仓库、记录服务器与 GPU/Python/CUDA 环境。

### Experiments

未启动模型 evaluation；本轮仅执行项目初始化和只读环境检查。

### Key Results

- 10 张 RTX 2080 Ti 在宿主环境均可见且空闲。
- CUDA Toolkit 12.8、GCC、Conda 和 Git 可用。
- 当前 Conda base 的用户级 PyTorch 安装不完整，不可用于复现。

### Problems

- baseline 官方仓库、固定 commit 与 checkpoint 尚未确认。
- 工作区磁盘仅剩约 55 GiB，下载 checkpoint/data 前需估算空间。
- 需为 baseline 新建隔离 Conda 环境并完成 CUDA smoke test。

### Decisions

不修补 base 环境；下一阶段依据 baseline 官方依赖创建隔离环境。继续遵循先复现、后改单变量的实验顺序。

### Git

初始化提交：`7594e3f2797c92358e60fdb935f133397a7aead0`

### Next

确认并固定 Fast-WAM / JointWAM 官方 baseline 仓库，阅读 README、evaluation 入口和 checkpoint 获取流程。

### Baseline acquisition and code audit

- Official repository cloned at `repos/FastWAM` and pinned to `7faa71108368fbb3b6885649f112af607427a2d4`.
- Evaluation entry points and inference paths were inspected without changing code.
- `E = EVALUATION.replan_steps` and `D = EVALUATION.num_inference_steps` are confirmed.
- H is derived from video-frame configuration but currently coupled to the default action horizon.
- W has no native multi-rollout implementation.
- Base-only assets require 23.129 GiB; the safe local peak budget is approximately 51.1 GiB against 54.58 GiB free.
- Official single-device loading is a GPU-memory blocker on 11 GiB RTX 2080 Ti hardware.
- Evidence: `notes/fastwam_baseline_audit_2026-09-13.md` and `notes/fastwam_asset_budget_2026-09-13.md`.
