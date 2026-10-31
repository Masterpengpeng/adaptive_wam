# 当前实验阶段

> 本文件是“当前处于计划表哪个阶段”的唯一状态页。开始新阶段时更新本页；已完成阶段的历史细节追加到 [实验进度日志](./experiment_progress.md)。

**更新时间：** 2026-10-14  
**当前阶段：** Phase 1 — 基础建设与真实模型硬件测速  
**当前子阶段：** EGL 双相机检查与 base checkpoint 单 Episode 闭环 smoke test 已完成；正在补齐 profiling 复测与多任务/多 seed 基线评测。  
**整体状态：** 进行中（profiling 数据可用，但尚不能称为完整论文复现）。

## 阶段计划与状态

| 子阶段 | 状态 | 已有证据 / 验收条件 |
|---|---|---|
| P1.1 环境与资产准备 | 已完成 | `.conda/fastwam` 环境；Optional-IDM checkpoint、匹配的 dataset stats、Wan T5 资产已下载并通过记录中的校验。 |
| P1.2 真实推理路径接入 | 已完成（基础验证） | 四卡 FP16 Optional-IDM 模型加载成功；`H=0` / `H=2` 均返回有限的 `(10,7)` 动作；最大网格 `H=8,D=16,W=4` 在 4 卡上成功运行。 |
| P1.3 全网格硬件 Profiling | 已完成一轮 | 37 组 H/D/W 写入 [真实成本表](../../results/phase2/real_fastwam_phase2_2080ti/cost_table.json)，FP16、RTX 2080 Ti、同步墙钟计时。 |
| P1.4 复测与结果验收 | 进行中 | 尚需独立重复 run，量化波动，并核对所有配置无缺行、异常值或 OOM。 |
| P1.5 官方/项目基线任务评测 | 单 Episode smoke 已完成；正式评测待开始 | Base checkpoint 在 LIBERO Spatial task 0、seed 0、H=0/E=10 下成功完成 77 步，Return=1.0；仅验证 rollout 闭环，不作为成功率基线。 |
| P1.6 EGL 环境与双相机渲染 | 已完成（task 0 smoke） | EGL 下两路图像均为非空 `224×224×3 uint8`；PNG、并排预览和 JSON 检查结果已保存。 |
| P2 Counterfactual 与 Oracle | 未开始 | 暂无真实 Counterfactual 结果、Oracle 对比或 Go/No-Go 结论。 |

## 当前阶段目标

完成真实模型 profiling 的可复现性检查，并建立可作为后续 H/D/W/E 对比基准的 LIBERO baseline 评测。Phase 1 验收前，不把成本表单独解释成策略效果或论文结论。

## 下一步

1. 用新的输出路径重复 37 配置 profiling，不覆盖已保存的结果。
2. 对代表配置至少重复若干次，报告均值/中位数、离散程度、温度/显存条件；保持相同 checkpoint、GPU 组、任务输入、FP16 和计时方法。
3. 确认成本表的 `H=0` 语义：网格中 `D=W=0` 是规范化表示，实际 direct path 使用 `direct_path_denoising_steps=10`。
4. 对照 FastWAM 官方评测设置运行固定任务集和多个 seed 的 base-checkpoint baseline，保存 episode 结果与成功率汇总。
5. 将成功率、Return、Latency 以及命令/参数回填到 [结果页](./experiment_results.md) 和 [进度日志](./experiment_progress.md)。

## Phase 1 完成门槛

- [ ] 37 个配置的独立复测完成；异常配置及重试均有记录。
- [ ] 成本测量方法、输入状态、GPU 组、FP16、warmup/iteration 数明确且可重复。
- [ ] GFLOPs 标明为模块 hook 估算，注明未覆盖的算子；不作为完整 FLOPs 计数器数据。
- [ ] LIBERO baseline 的任务集、任务数、seed 数、成功率与 Return 有原始数据支撑。
- [ ] README/实验记录准确引用最终成本表与 baseline 输出目录。
- [x] 单 task EGL 双相机渲染、base-checkpoint 完整 episode 闭环及 MP4 解码检查。

## 重要边界

- 成本表只描述当前真实 Optional-IDM checkpoint 和 profiling 输入，不等同于多任务策略成功率实验。
- 表中 GFLOPs 是 Linear/Conv2d 模块 forward-hook MAC 换算的估算，遗漏 functional attention 等运算。
- `H=0` 的 direct action 仍使用 10 个 action denoising steps；不能把 H=0 行解释为零步去噪。
- 单个 task/seed 的性能不能证明统计显著性或跨任务泛化。
- Phase 2 的 state-wise Oracle 结果如果使用同一批状态选择并评估，属于 in-sample 上界；泛化判断需 held-out states/seeds。
