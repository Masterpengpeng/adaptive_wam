# 实验进度日志

本文件只追加已发生、可核验的事件。当前计划阶段与待办请看[当前阶段](./current_stage.md)；结果解读请看[实验效果与结果](./experiment_results.md)。

## 2026-10-14 — Phase 1 真实模型 profiling 进展

- **目标：** 使用实际 FastWAM Optional-IDM 权重验证推理接口，并测量 H/D/W 组合成本。
- **资产：** `libero_optional_idm_2cam224.pt`、匹配的 `libero_optional_idm_2cam224_dataset_stats.json` 和 Wan T5 encoder 已下载。下载过程中的 T5 首次 hash mismatch 未被使用；清除损坏 partial 后重新下载，最终通过 manifest SHA-256 校验。
- **实现修复：**
  - Adapter 根据 checkpoint 文件名解析匹配的 dataset stats。
  - Hydra 覆盖改为真实 schema 中的 `EVALUATION.device`。
  - DiffSynth 的本地模型根目录由 `DIFFSYNTH_MODEL_BASE_PATH` 显式设置。
  - LIBERO 初始化状态的空值检查改为长度检查，以兼容 NumPy 数组。
  - profiling/collection 默认设置 expandable CUDA allocator，减少大配置显存分配碎片风险。
- **真实推理检查：** 4 张 RTX 2080 Ti、FP16；LIBERO `libero_spatial` task 0、seed 0。`H=0,D=0,W=0,E=10` 与 `H=2,D=2,W=1,E=10` 返回有限的 `(10,7)` 动作。最大配置 `H=8,D=16,W=4,E=10` 在清理缓存后成功运行；记录峰值分配约为 GPU0 9.81 GiB、GPU1 3.54 GiB、GPU2 6.30 GiB、GPU3 4.34 GiB。
- **Profiling：** 输出 [真实成本表](../../results/phase2/real_fastwam_phase2_2080ti/cost_table.json)，确认包含 37 组配置；FP16，RTX 2080 Ti，10 warmup、50 次计时样本。多卡延迟为所有模型设备同步后的端到端墙钟时间；GFLOPs 为 Linear/Conv2d hook 估算。
- **代表结果：** `H=0` 约 931.30 ms；`H=2,D=2,W=1` 约 488.32 ms；`H=2,D=16,W=4` 约 12.13 s；`H=8,D=16,W=4` 约 24.89 s。详见[结果页](./experiment_results.md)。
- **测试：** 项目 Python 3.10 环境下 `python -m unittest discover -s tests`：8 项通过。相关 Python 文件的 Pylance diagnostics 无问题。
- **解释限制：** 该成本表没有 LIBERO success rate 或多 seed 结果；H=0 使用 10 个 direct-path action denoising steps。首次最大配置单独探针曾因显存碎片 OOM；设置 expandable allocator 并按正式 profiler 清缓存后最大网格推理通过。
- **本次未完成：** 独立重复 profiling、FastWAM 官方设置下的 LIBERO baseline 成功率评测、完整 Phase 2 counterfactual 扫描及 Oracle 分析。

## 2026-10-14 — EGL 双相机与 base-checkpoint rollout smoke

- **目标：** 完成阶段计划中尚缺的 EGL 双目渲染检查和单个 LIBERO episode 闭环。
- **新增入口：** `scripts/check_env.py` 做不加载策略权重的 task/EGL/相机检查；`eval_single_rollout.py` 加载真实 FastWAM 权重、推理并执行动作 chunk、保存视频与 episode JSON。两者均提供参数并拒绝覆写非空输出目录。
- **执行命令：**
  - `python scripts/check_env.py --task-suite libero_spatial --task-id 0 --seed 0 --resolution 224 --output-dir outputs/phase1/env_smoke_20261014`
  - `python eval_single_rollout.py --model_path checkpoints/fastwam_release/libero_uncond_2cam224.pt --task-config libero_uncond_2cam224_1e-4 --task-suite libero_spatial --task-id 0 --seed 0 --H 0 --D 0 --W 0 --E 10 --max-steps 400 --gpu-ids 0,1,2,3 --output-dir outputs/phase1/baseline_rollout_task0_seed0_20261014`
- **环境检查：** `MUJOCO_GL=egl`，LIBERO `libero_spatial` task 0、seed 0、224 分辨率。`agentview_image` 与 `robot0_eye_in_hand_image` 均为非空 `(224,224,3) uint8`。PNG 与 `env_check.json` 保存于 `outputs/phase1/env_smoke_20261014/`。
- **指定 baseline rollout：** checkpoint `libero_uncond_2cam224.pt`，task config `libero_uncond_2cam224_1e-4`，4 张 RTX 2080 Ti、FP16，H=0/E=10，seed 0。Episode 成功结束：Return=1.0、77 环境步、8 次策略调用、平均推理时间 1120.19 ms；视频共 78 帧（含初始观测），双相机拼接帧尺寸 `(256,512,3)`。
- **产物：** `outputs/phase1/baseline_rollout_task0_seed0_20261014/episode.json` 与 `episode.mp4`。MP4 已解码验证 78 帧，首尾帧尺寸正确。另一次 Optional-IDM checkpoint rollout 也成功，但不作为 base-checkpoint baseline 统计。
- **解释限制：** 单 task/seed 的成功只证明环境到模型到动作执行再到录像的闭环可运行，不是 LIBERO 总体成功率。正式 baseline 仍需遵循 FastWAM 官方任务数与 trial 数，并聚合 episode-level 指标。
- **验证：** 环境脚本输出 status=`passed`；base rollout 输出 success=`true`；项目 8 项单元测试通过；Pylance 在选定项目解释器后无诊断。

### 下一次记录模板

```text
## YYYY-MM-DD — <阶段 / 实验名>

- Run ID / commit:
- 目标:
- 环境 / GPU:
- checkpoint / config:
- 命令与参数:
- 输出目录:
- 结果:
- 失败、重试或偏差:
- 结论及其适用范围:
- 下一步:
```
