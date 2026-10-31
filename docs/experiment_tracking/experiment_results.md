# 实验效果与结果汇总

> 只记录从原始结果文件中核验的数据。当前包含 Phase 1 硬件成本测量和单 episode 闭环 smoke test；后者不是 benchmark 成功率估计。

## 当前可用数据

### 真实 FastWAM 成本表

- **结果文件：** [cost_table.json](../../results/phase2/real_fastwam_phase2_2080ti/cost_table.json)
- **记录数：** 37 个 H/D/W 配置。
- **设备：** NVIDIA GeForce RTX 2080 Ti；FastWAM 放置在 4 卡副本上（profiling 使用 GPU 0–3）。
- **精度：** FP16。
- **计时：** 多设备同步后的端到端墙钟时间；每组 10 次 warmup、50 次计时样本。
- **模型：** 本地验证过的 `libero_optional_idm_2cam224.pt`，匹配 dataset stats。
- **H=0 约定：** 表内 `H=0,D=0,W=0`；实际 direct action denoising steps 记录为 10。
- **GFLOPs：** Linear 与 Conv2d forward hook 统计模块 MAC 并按每 MAC 两 FLOPs 换算。Functional attention matmul、部分自定义算子与 elementwise 运算未被完整计入，因此是估算值。

### 代表配置

| H | D | W | Latency (ms) | GFLOPs（估算） | 说明 |
|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 931.30 | 2,049.78 | direct path；实际 action denoising steps = 10 |
| 2 | 2 | 1 | 488.32 | 8,446.40 | 当前表格中较轻的 imagination 配置 |
| 2 | 16 | 4 | 12,125.34 | 193,184.03 | 高 D/W 配置 |
| 8 | 16 | 4 | 24,890.68 | 545,723.48 | H/D/W 最大网格点 |

完整 37 行数值以 JSON 文件为准；不要手工从本表反推或重建成本表。

## 单 episode 环境闭环 smoke test

| Checkpoint | Task / seed | 配置 | Episode length | Return | Success | 推理调用 | 平均推理 latency |
|---|---|---|---:|---:|---|---:|---:|
| `libero_uncond_2cam224.pt` | LIBERO Spatial task 0 / seed 0 | H=0, D=0, W=0, E=10；direct depth=10 | 77 | 1.0 | 是 | 8 | 1120.19 ms |

- episode JSON：[episode.json](../../outputs/phase1/baseline_rollout_task0_seed0_20261014/episode.json)
- episode MP4：[episode.mp4](../../outputs/phase1/baseline_rollout_task0_seed0_20261014/episode.mp4)，已解码检查 78 帧，拼接双相机帧尺寸为 256×512 RGB。
- EGL 渲染检查：[env_check.json](../../outputs/phase1/env_smoke_20261014/env_check.json)，两路相机原图和并排图在同一目录。
- **结论边界：** 这是 seed 0、单 task 的闭环 smoke test；一次成功不构成 benchmark 成功率估计或论文复现结果。

## 初步观察（非任务效果结论）

1. 增加 D 和 W 会显著增加推理耗时与估算计算量；最大 H/D/W 配置比轻量配置贵得多。
2. `H=0` 行采用 10 步 direct action denoising，而 H>0 行的 D 字段表示 IDM 推理深度。因此不能把 `H=0,D=0` 解读为比任意 H>0 配置都少去噪，也不能在不注明该语义的情况下直接比较 D。
3. `H=2,D=2,W=1` 的实测延迟低于当前 H=0 行，可能与 direct path 固定 10 步、实际模型分支和运行时开销有关；应按真实推理配置解释，不能只按 H 大小排序。
4. 目前数据对应单一 LIBERO task/seed 的 profiling observation，且每配置为一轮成本测量；它支持硬件成本初步评估，但不代表平均策略行为或最终上线延迟保证。

## 尚未获得的任务效果指标

| 指标 | 状态 | 需要的实验 |
|---|---|---|
| LIBERO 成功率 | 尚无真实基线结果 | 固定官方评测设置、任务集与多个 seed 评估 |
| 每集 Return | 尚无正式对照 | 保存 episode-level reward 与 termination |
| E 对成功率/成本的影响 | 尚未测量 | 固定 checkpoint/task/seeds，单独 sweep execution horizon |
| H/D/W 对任务成功率的影响 | 尚未测量 | 在固定 E 和一致的初始状态上做受控对比 |
| State-wise adaptation 增益 | 尚无 Phase 2 结果 | 完成 counterfactual 采集，并使用 held-out state/seed 评估 |
| 统计不确定性 | 尚未计算 | 以 episode/seed 为单位报告置信区间或 bootstrap 区间 |

## 后续报告规范

- 报告 success rate 时同时给出成功数、episode 总数、任务数和 seed 数。
- 报告 latency 时声明是否包含预处理、同步、模型输出后处理和多卡通信，并给出样本分布而非只给均值。
- 成本/效果比较必须使用同一设备、同一 checkpoint、相同任务与 seed 设计。
- Oracle 若在同一批 rollout 上选择配置并报告效果，标注为 in-sample oracle；不得据此宣称统计显著或泛化。
