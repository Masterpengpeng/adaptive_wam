# Research Context

> Updated: 2026-09-13 12:28 +08:00
> Evidence through: baseline code audit and metadata-only asset budget
> Canonical record: [`../EXPERIMENT_LOG.md`](../EXPERIMENT_LOG.md)

This file is a replaceable current-state snapshot for quickly resuming work. Formal
experiment results and complete handoffs belong in `EXPERIMENT_LOG.md` after review
by the main agent.

## Now

- Stage: baseline acquired; isolated-environment preparation
- Active experiment: none (`EXP-B000` has not started)
- Objective: create the official Python environment and run no-weight CUDA/BF16 checks
- Status: ready for environment creation; model evaluation blocked by GPU memory
- Owner: main agent / `experiment_runner`

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
- Do not overwrite outputs; before long runs, report GPU count, exact command, and output path
- Estimate repository, checkpoint, dataset, and output disk usage before downloading artifacts

## Last Verified State

- Project HEAD before current documentation updates: `df0177b` (`Add resumable research context snapshot`)
- Worktree: documentation and baseline-lock updates pending review
- Baseline checkout: clean; approximately 11 MiB
- Required environment: Python 3.10, `torch 2.7.1+cu128`, `torchvision 0.22.1+cu128`
- Local environment: dedicated Conda environment not created; base Python has an incomplete
  user-level `torch 2.13.0` installation and remains unsuitable for the baseline
- Hardware: 10 × RTX 2080 Ti (11,264 MiB each); CUDA toolkit 12.8
- Storage: 54.58 GiB free; base-only safe peak budget is approximately 51.1 GiB
- Evaluation entry points: manager at `experiments/libero/run_libero_manager.py`, which
  dispatches single evaluation through `experiments/libero/eval_libero_single.py`
- Latest experiment command: none
- Latest experiment output: none
- Result: project initialization (`EXP-000`) is complete; no training or evaluation has run

## Open Issues

- [x] Determine checkpoint location, size, checksum/version, and access requirements using metadata only.
- [x] Determine the size of automatically downloaded VAE, T5, and tokenizer assets without downloading them.
- [ ] Resolve the single-device GPU-memory blocker before downloading model assets.
- [ ] Resolve the `num_frames` / `action_horizon` coupling before designing an `H` sweep.
- [ ] Characterize variant-dependent `num_inference_steps` compute before designing a `D` sweep.
- [ ] Decide whether an evidence-preserving implementation extension is justified for `W`,
      which the baseline does not support.
- [ ] Create the isolated Python 3.10 environment and run no-weight CUDA/BF16 smoke tests.
- [ ] Recheck free space and prepare a download budget before fetching checkpoints or data.

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
- Keep following `DEC-001` through `DEC-003`: Oracle before Controller, one axis at a time,
  and public-checkpoint inference/evaluation before large-model training.

## Next Exact Action

1. Create an isolated Python 3.10 environment without downloading model assets.
2. Install the pinned official software dependencies with caches kept inside the project.
3. Run no-weight PyTorch/CUDA compute-capability and BF16 smoke tests.
4. Do not download checkpoints until an offload/sharding or larger-memory GPU plan is verified.

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
- `2026-09-13` — `notes/environment_2026-09-13.md` — base PyTorch import fails and the
  workspace filesystem had approximately 55 GiB free.
