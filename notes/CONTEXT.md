# Research Context

> Updated: 2026-09-13 12:37 +08:00
> Evidence through: isolated core-environment installation and 10-GPU dtype smoke-test report
> Canonical record: [`../EXPERIMENT_LOG.md`](../EXPERIMENT_LOG.md)

This file is a replaceable current-state snapshot for quickly resuming work. Formal
experiment results and complete handoffs belong in `EXPERIMENT_LOG.md` after review
by the main agent.

## Now

- Stage: isolated core environment validated; offload/sharding feasibility study
- Active experiment: none (`EXP-B000` has not started)
- Objective: find a semantics-preserving way to evaluate the model within available GPU memory
- Status: core environment GO; model evaluation NO-GO on a single 11 GiB GPU
- Owner: `experiment_runner`

## Fixed / Do Not Change

- Baseline repo: `repos/FastWAM`
- Origin: <https://github.com/yuantianyuan01/FastWAM.git>
- Baseline commit: `7faa71108368fbb3b6885649f112af607427a2d4`
- Checkpoint: TBD
- Benchmark: LIBERO; tasks and seeds TBD
- `E`: `replan_steps`
- `D`: `num_inference_steps`; compute behavior is variant-dependent and must be measured per variant
- `H`: derived from `num_frames`, but coupled to `action_horizon`; do not sweep it as an
  independent axis until that coupling is resolved
- `W`: unsupported by the audited baseline
- Do not change more than one supported axis in an experiment
- Do not use or repair the base Python environment for this project
- Use `PYTHONNOUSERSITE=1` and keep package/model caches inside the project
- Do not overwrite outputs; before long runs, report GPU count, exact command, and output path
- Estimate repository, checkpoint, dataset, and output disk usage before downloading artifacts

## Last Verified State

- Project HEAD before current smoke-test documentation: `e4b31ff` (`Audit FastWAM baseline and asset requirements`)
- Worktree: smoke-test documentation updates pending review
- Baseline checkout: clean; approximately 11 MiB
- Isolated prefix: `.conda/fastwam`
- Validated core environment: Python 3.10.21, `torch 2.7.1+cu128`,
  `torchvision 0.22.1+cu128`, CUDA 12.8; `pip check` clean
- Isolation/cache policy: `PYTHONNOUSERSITE=1`; caches are configured inside the project
- Hardware: 10 × RTX 2080 Ti (11,264 MiB each), compute capability 7.5
- Dtype smoke test: FP16 and BF16 each produced finite results on all 10 GPUs (`20/20`);
  per dtype and per GPU peak allocated memory was 8,564,736 B and peak reserved memory was
  23,068,672 B
- Storage: environment plus project-local caches use 8.00 GiB; 46.58 GiB remains free
- Evaluation entry points: manager at `experiments/libero/run_libero_manager.py`, which
  dispatches single evaluation through `experiments/libero/eval_libero_single.py`
- Latest validation: no-weight FP16/BF16 smoke test across all 10 GPUs; no model evaluation
- Latest experiment output: none
- Result: core environment is GO; checkpoint-backed model evaluation has not run and remains
  NO-GO because official single-device loading exceeds per-GPU memory

## Open Issues

- [x] Determine checkpoint location, size, checksum/version, and access requirements using metadata only.
- [x] Determine the size of automatically downloaded VAE, T5, and tokenizer assets without downloading them.
- [ ] Resolve the single-device GPU-memory blocker before downloading model assets.
- [ ] Verify whether offload or sharding can preserve official algorithm and evaluation semantics.
- [ ] Recompute disk requirements for the selected offload/sharding path before further installs.
- [ ] Resolve the `num_frames` / `action_horizon` coupling before designing an `H` sweep.
- [ ] Characterize variant-dependent `num_inference_steps` compute before designing a `D` sweep.
- [ ] Decide whether an evidence-preserving implementation extension is justified for `W`,
      which the baseline does not support.
- [x] Create the isolated Python 3.10 core environment and run no-weight FP16/BF16 smoke tests.
- [ ] Install remaining dependencies and weights only after the feasibility and disk reviews pass.

## Decisions Since Last Handoff

- The official FastWAM repository is cloned and pinned at
  `7faa71108368fbb3b6885649f112af607427a2d4`; keep the baseline checkout clean.
- Record `E = replan_steps`; record `D = num_inference_steps` with a variant-dependent compute
  qualification; record `H` as `num_frames`-derived but coupled to `action_horizon`.
- Record `W` as unsupported rather than inventing a parameter mapping.
- Automatic VAE, T5, and tokenizer downloads are a disk-risk item and require a metadata-only
  size budget before environment setup or evaluation.
- Base-only model assets total 23.129 GiB; base plus Optional-IDM total 34.344 GiB.
- Base-only storage is a conditional GO, but official single-device evaluation is a NO-GO on
  11 GiB GPUs until an offload, sharding, or larger-memory-device path is verified.
- The isolated core environment at `.conda/fastwam` is a GO: dependency consistency is clean,
  and both tested dtypes produced finite results on every GPU. This does not clear the model
  evaluation memory blocker.
- Keep following `DEC-001` through `DEC-003`: Oracle before Controller, one axis at a time,
  and public-checkpoint inference/evaluation before large-model training.

## Next Exact Action

1. Inspect semantics-preserving offload and sharding options for the pinned baseline without
   changing the model algorithm or downloading weights.
2. Estimate device placement, per-device memory, host-memory, and any additional disk/cache
   requirements for viable options.
3. Recompute available disk headroom against the chosen path.
4. Only after both reviews pass, install the remaining dependencies and download required weights.

## Recent Evidence

- `2026-09-13 12:11 +08:00` — local Git inspection — branch `main`, clean worktree,
  HEAD `6c805bf870d67e59f37d3e17ef74af89f5d5581c` before this file was added.
- `2026-09-13 12:21 +08:00` — `experiment_runner` acquisition report — official FastWAM clone
  at `repos/FastWAM`, origin <https://github.com/yuantianyuan01/FastWAM.git>, clean checkout at
  `7faa71108368fbb3b6885649f112af607427a2d4`, approximately 11 MiB.
- `2026-09-13 12:21 +08:00` — `experiment_runner` code-audit report — Python 3.10 with
  `torch 2.7.1+cu128` and `torchvision 0.22.1+cu128`; manager and single LIBERO evaluation
  entry points verified; `E = replan_steps`, `D = num_inference_steps` with variant-dependent
  compute, `H` is `num_frames`-derived and coupled to `action_horizon`, and `W` is unsupported.
- `2026-09-13 12:21 +08:00` — `experiment_runner` code-audit report — evaluation may
  automatically download VAE, T5, and tokenizer assets, requiring a metadata-only size audit.
- `2026-09-13 12:28 +08:00` — Hugging Face/ModelScope metadata audit — base-only required
  assets total 23.129 GiB; safe peak storage budget is approximately 51.1 GiB against 54.58 GiB
  free; official one-device loading remains a GPU-memory NO-GO on RTX 2080 Ti.
- `2026-09-13 12:37 +08:00` — `experiment_runner` environment report — `.conda/fastwam`
  contains Python 3.10.21, `torch 2.7.1+cu128`, and `torchvision 0.22.1+cu128` with CUDA 12.8;
  `PYTHONNOUSERSITE=1`, project-local caches, and a clean `pip check` were verified.
- `2026-09-13 12:37 +08:00` — `experiment_runner` 10-GPU smoke-test report — all FP16 and
  BF16 checks were finite (`20/20`); each dtype/GPU run peaked at 8,564,736 B allocated and
  23,068,672 B reserved on compute-capability-7.5 devices.
- `2026-09-13 12:37 +08:00` — `experiment_runner` storage report — environment plus caches
  uses 8.00 GiB and 46.58 GiB remains; core environment is GO, while model evaluation remains
  NO-GO due to single-GPU memory.
- `2026-09-13` — `notes/environment_2026-09-13.md` — base PyTorch import fails and the
  workspace filesystem had approximately 55 GiB free.
