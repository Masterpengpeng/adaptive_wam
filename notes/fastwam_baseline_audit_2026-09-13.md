# FastWAM Baseline Audit — 2026-09-13

## Provenance

- Repository: <https://github.com/yuantianyuan01/FastWAM>
- Local path: `repos/FastWAM`
- Branch: `main`
- Commit: `7faa71108368fbb3b6885649f112af607427a2d4`
- Commit subject: `Optimize IDM action-only inference`
- License: MIT
- Clone size: approximately 11 MiB
- Submodules/LFS payload: none detected

The repository was cloned with:

```bash
git clone https://github.com/yuantianyuan01/FastWAM.git repos/FastWAM
```

No dependency, checkpoint, base-model asset, or dataset was downloaded during this audit.

## Official environment

- Python 3.10
- PyTorch 2.7.1+cu128
- torchvision 0.22.1+cu128
- FastWAM dependencies are declared in `pyproject.toml`
- LIBERO is installed separately; FastWAM requires `mujoco==3.3.2`

The existing Conda base environment must not be reused.

## Evaluation path

- Multi-task manager: `experiments/libero/run_libero_manager.py`
- Single-task worker: `experiments/libero/eval_libero_single.py`
- Main evaluation config: `configs/sim_libero.yaml`
- Default evaluation is large: four LIBERO suites, eight GPUs, and 50 trials per task.
- The first smoke test must instead use one GPU, one task, one trial, and a fixed seed.

High-level action path:

```text
Hydra config
  -> run_libero_manager.py
  -> eval_libero_single.py
  -> instantiate model/runtime and load checkpoint/stats
  -> observe two cameras + proprioception
  -> _predict_action_chunk
  -> model.infer_action
  -> unnormalize actions
  -> enqueue first replan_steps actions
  -> execute one action per simulator step
  -> reobserve and replan when the queue is empty
```

## H / D / W / E mapping

| Axis | Code mapping | Confidence | Important qualification |
|---|---|---:|---|
| E | `EVALUATION.replan_steps` | High | Controls how many generated actions are executed before replanning; it is not `action_horizon`. |
| D | `EVALUATION.num_inference_steps` | High | Base uses D action steps; Joint uses D joint video/action steps; IDM uses D video plus D action steps, so equal numeric D does not imply equal compute. |
| H | `data.train.num_frames` and `action_video_freq_ratio` determine video frames | Medium-high | Currently coupled to the default action horizon. Keep `EVALUATION.action_horizon` fixed before an H sweep. Native no-imagination comparison is Optional-IDM `first_frame`, not an arbitrary one-frame substitution. |
| W | No native configuration | High | Evaluation calls inference once; batch/noise paths assume batch size one. W>1 needs explicit outer-loop seeds and a separately defined selection rule. |

## Model variants

- FastWAM base: first-frame-only action denoising; no generated future video.
- JointWAM: jointly denoises future-video and action latents.
- IDM: first denoises future video, then freezes it and denoises actions using full-video KV cache.
- Optional-IDM: one checkpoint can select `idm` or `first_frame`, making it the cleanest native imagination/no-imagination comparison.

Confirmed public checkpoint pairs:

- `libero_uncond_2cam224.pt` plus matching dataset statistics; official reproduction uses `sigma_shift=5.0`.
- `libero_optional_idm_2cam224.pt` plus matching dataset statistics; official reproduction uses `sigma_shift=1.0`.

Independent Joint/IDM checkpoints were not confirmed in the published file list and must not be assumed available.

## Blockers before evaluation

1. The loader can automatically download VAE, UMT5-XXL, and tokenizer assets. Their exact sizes must be checked through metadata before model instantiation.
2. Only about 55 GiB is free on the workspace filesystem.
3. The current runtime moves the roughly 5B model to one CUDA device without an evident device-map, sharding, or CPU-offload path. A single 11 GiB RTX 2080 Ti is likely insufficient.
4. Default BF16 and compiled inference require an explicit Turing/sm75 compatibility smoke test.
5. Cold compilation latency and warm inference latency must be recorded separately.
6. H is coupled to action horizon, and W requires new sampling and selection semantics; neither should be swept yet.

## Next action

Use Hugging Face and ModelScope metadata only—without loading the model—to compute the minimum download and disk-space budget. Decide whether this host can support the official checkpoint before creating the environment or requesting model downloads.
