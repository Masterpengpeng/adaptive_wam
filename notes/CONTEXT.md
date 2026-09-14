# Research Context

> Updated: 2026-09-14 08:52 +08:00
> Evidence through: multi-GPU implementation, dependency, EGL, and launcher preflight reports
> Canonical record: [`../EXPERIMENT_LOG.md`](../EXPERIMENT_LOG.md)

This file is a replaceable current-state snapshot for quickly resuming work. Formal
experiment results and complete handoffs belong in `EXPERIMENT_LOG.md` after review
by the main agent.

## Now

- Stage: base-only asset acquisition readiness
- Active experiment: `EXP-B000`
- Objective: download and verify the audited base-only assets, then run one four-GPU trial
- Status: software and no-model preflight PASS; no model result yet
- Owner: `experiment_runner`

## Fixed / Do Not Change

- Baseline repo: `repos/FastWAM`
- Origin: <https://github.com/yuantianyuan01/FastWAM.git>
- Baseline commit: `7faa71108368fbb3b6885649f112af607427a2d4`
- Adaptation branch: `exp/multigpu-placement`
- Adaptation commits: `f60e17c`, `47baac4`, `ead553d`
- Checkpoint/assets: base-only set, 23.129 GiB; not yet downloaded or validated
- Benchmark: LIBERO; tasks and seeds TBD
- LIBERO commit: `8f1084e`
- LIBERO environment policy: do not install its legacy `requirements.txt`; use the compatible
  minimal stack and an explicit `PYTHONPATH`
- `E`: `replan_steps`
- `D`: `num_inference_steps`; compute behavior is variant-dependent and must be measured per variant
- `H`: derived from `num_frames`, but coupled to `action_horizon`; do not sweep it as an
  independent axis until that coupling is resolved
- `W`: unsupported by the audited baseline
- Do not change more than one supported axis in an experiment
- There is no configuration-only solution for official evaluation on the available 11 GiB GPUs
- Keep `compile=false` for the proposed multi-GPU path
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
- Core and FastWAM dependencies: installed successfully
- Isolation/cache policy: `PYTHONNOUSERSITE=1`; caches are configured inside the project
- Hardware: 10 × RTX 2080 Ti (11,264 MiB each), compute capability 7.5
- Dtype smoke test: FP16 and BF16 each produced finite results on all 10 GPUs (`20/20`);
  per dtype and per GPU peak allocated memory was 8,564,736 B and peak reserved memory was
  23,068,672 B
- Storage: approximately 45 GiB remains free before the base-only asset download
- Audited BF16 component sizes: MoT has 6,020,710,599 parameters (approximately 11.214 GiB),
  T5 is approximately 10.582 GiB, and VAE is approximately 1.313 GiB
- Evaluation entry points: manager at `experiments/libero/run_libero_manager.py`, which
  dispatches single evaluation through `experiments/libero/eval_libero_single.py`
- Multi-GPU tests: `20/20` PASS across commits `f60e17c`, `47baac4`, and `ead553d`
- Latest simulation validation: no-model EGL smoke PASS on `libero_spatial` task 0 with fixed
  initial state, one step, and two 64×64 renders; total time 6.49 s
- Asset/launcher preflight: launcher tests `4/4` PASS; compose-only PASS
- Latest experiment output: none
- Result: implementation and software preflight are ready for asset acquisition. No checkpoint
  has been loaded and no model evaluation result exists; performance preservation or efficiency
  improvement must not be claimed

## Open Issues

- [x] Determine checkpoint location, size, checksum/version, and access requirements using metadata only.
- [x] Determine the size of automatically downloaded VAE, T5, and tokenizer assets without downloading them.
- [x] Audit offload/sharding feasibility and confirm that no configuration-only solution exists.
- [x] Choose option A (official path on a GPU with at least 24/32 GB) or authorize option B
      (multi-GPU non-algorithm patch on a separate baseline branch). Option B selected.
- [x] Implement and test the multi-GPU placement path (`20/20` PASS).
- [x] Install the core, FastWAM, and compatible minimal LIBERO software stack.
- [x] Recompute disk headroom: approximately 45 GiB free for the 23.129 GiB base-only set.
- [ ] Download and verify the base-only assets.
- [ ] Run one four-GPU trial after asset verification.
- [ ] Resolve the `num_frames` / `action_horizon` coupling before designing an `H` sweep.
- [ ] Characterize variant-dependent `num_inference_steps` compute before designing a `D` sweep.
- [ ] Decide whether an evidence-preserving implementation extension is justified for `W`,
      which the baseline does not support.
- [x] Create the isolated Python 3.10 core environment and run no-weight FP16/BF16 smoke tests.
- [ ] Do not report performance or efficiency until checkpoint-backed evaluation produces results.

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
- Preferred multi-GPU adaptation, if authorized: keep `compile=false`; cache task prompts;
  pre-encode T5 across two GPUs and then offload it to CPU; place the video expert on GPU 0 and
  action expert plus VAE on GPU 1. The audited KV transfer is approximately 34 MiB.
- If the video GPU lacks working-memory headroom, use paired-layer sharding of the video expert
  across two GPUs. Sequential offload is a fallback only.
- Accelerate device-map/FSDP paths are not recommended for this inference architecture.
- The multi-GPU non-algorithm placement implementation is fixed by commits `f60e17c`,
  `47baac4`, and `ead553d`; its current structural/device test suite passes `20/20`.
- LIBERO is fixed at `8f1084e`. Its legacy requirements are forbidden because they conflict
  with the validated stack; use the compatible minimal dependencies and explicit `PYTHONPATH`.
- Preserve the first failed editable-packaging import attempt as setup evidence. The subsequent
  no-model EGL smoke passed with the explicit import path.
- Keep following `DEC-001` through `DEC-003`: Oracle before Controller, one axis at a time,
  and public-checkpoint inference/evaluation before large-model training.

## Next Exact Action

1. Download the audited base-only 23.129 GiB asset set without optional IDM assets.
2. Verify every downloaded asset against its expected identity, size, and checksum before use.
3. Run exactly one four-GPU trial through the validated launcher.
4. Record model output, peak memory, latency, failures, and artifact paths before expanding runs.

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
- `2026-09-13 12:46 +08:00` — `experiment_runner` multi-GPU feasibility audit — no
  configuration-only solution exists. BF16 sizes are approximately 11.214 GiB for the
  6,020,710,599-parameter MoT, 10.582 GiB for T5, and 1.313 GiB for VAE.
- `2026-09-13 12:46 +08:00` — `experiment_runner` multi-GPU feasibility audit — recommended
  adaptation is `compile=false`, task-prompt caching, two-GPU T5 pre-encoding followed by CPU
  offload, video expert on GPU 0, and action expert plus VAE on GPU 1, with approximately
  34 MiB KV transfer; paired-layer video sharding is the low-headroom alternative.
- `2026-09-13 12:46 +08:00` — `experiment_runner` multi-GPU feasibility audit — sequential
  offload is fallback-only; Accelerate device-map and FSDP are not recommended.
- `2026-09-14 08:52 +08:00` — `experiment_runner` implementation report — FastWAM branch
  `exp/multigpu-placement` at commits `f60e17c`, `47baac4`, and `ead553d`; multi-GPU tests
  passed `20/20`.
- `2026-09-14 08:52 +08:00` — dependency report — core and FastWAM dependencies installed;
  LIBERO pinned at `8f1084e` with a compatible minimal stack and explicit `PYTHONPATH`, while
  its legacy requirements remain prohibited.
- `2026-09-14 08:52 +08:00` — setup/simulation report — the first editable-packaging import
  failed and was retained; the following no-model EGL smoke passed on `libero_spatial` task 0,
  fixed initial state, one step, two 64×64 renders, in 6.49 s.
- `2026-09-14 08:52 +08:00` — asset/launcher preflight — launcher tests passed `4/4` and
  compose-only passed; approximately 45 GiB remains. No model result exists yet.
- `2026-09-13` — `notes/environment_2026-09-13.md` — base PyTorch import fails and the
  workspace filesystem had approximately 55 GiB free.
