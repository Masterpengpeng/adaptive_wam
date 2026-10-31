# Adaptive Inference for World Action Models

This repository is the experiment workspace for studying state-adaptive inference configurations in World Action Models.

The four inference variables are:

- `H`: future prediction / imagination horizon
- `D`: denoising / refinement depth
- `W`: number of stochastic future rollouts
- `E`: execution horizon / action-chunk commitment length

## Current stage

Phase 2 tooling is implemented for counterfactual collection and oracle
analysis. Real FastWAM/LIBERO execution requires a compatible IDM checkpoint,
the repository's configured assets, and the pinned environment dependencies.

## Directory layout

- `repos/`: third-party baseline repositories
- `configs/`: versioned experiment configurations
- `scripts/`: evaluation, profiling, and oracle tooling
- `checkpoints/`, `data/`: local large artifacts (ignored by Git)
- `outputs/`, `logs/`: raw experiment artifacts (ignored by Git)
- `results/`: processed tables and figures
- `notes/`: environment and research notes

## Phase 1: inference profiling

`model_stub.py` provides a composable `WorldActionModel` interface. Its default
modules are a lightweight synthetic model for validating adaptive inference
control flow; they are not the pretrained FastWAM architecture or checkpoint.
Replace the encoder, future generator, and action head modules to profile a
real policy using the same interface.

Run `python profile_cost_table.py` from the project environment to profile the
full H/D/W grid and write `cost_table.json` in the project root. The default is
10 warmup calls and 50 samples per configuration, with FP16 autocast. The
reported GFLOPs estimate counts Linear and Conv2d module multiply-adds via
forward hooks; it omits functional attention matmuls, elementwise operations,
and random-number generation, so treat it as an estimate rather than a
hardware-counter measurement. Real multi-GPU latency is synchronized
end-to-end wall time; single-GPU latency uses CUDA events. Use `--warmup` and
`--iterations` to run a shorter smoke profile.

### Parameterized FastWAM hardware profile

For the real `ParameterizedWAM` path, run the two-camera FP16 hardware profile
with a checkpoint and four model GPUs:

```bash
./.conda/fastwam/bin/python profiling/profile_hardware.py \
  --model-path checkpoints/fastwam_release/libero_optional_idm_2cam224.pt \
  --gpu-ids 0,1,2,3 \
  --output configs/hardware_cost_table.json
```

This scans the action-only configuration and all 36 H/D/W world-model
configurations, with 10 warmups and 50 CUDA-event samples by default. It
generates synthetic 224x224 uint8 images for both LIBERO cameras, prints a
Markdown table, and writes the structured result to
`configs/hardware_cost_table.json`. `relative_compute` is the measured mean
latency divided by the Fixed-Full (H=8, D=16, W=4) latency. Run the smoke tests
without loading a checkpoint using:

```bash
./.conda/fastwam/bin/python -m unittest tests.test_profiling -v
```

### Phase 1 environment and rollout smoke tests

Run the three fixed LIBERO baselines (smoke default: Spatial task 0, two
seeds); use `--task-ids 0-4` for the full LIBERO-Spatial task set:

```bash
./.conda/fastwam/bin/python eval_fixed_baselines.py \
  --model-path checkpoints/fastwam_release/libero_optional_idm_2cam224.pt \
  --task-suite libero_spatial \
  --task-ids 0-4 \
  --num-seeds 2 \
  --gpu-ids 0,1,2,3
```

The script compares No-Imagination, Fixed-Small, and Fixed-Full from the same
saved initial state. It writes aggregate metrics to
`results/fixed_baselines_summary.csv` and per-episode records to
`results/fixed_baselines_episodes.csv`. Each method receives five virtual
forward warm-ups before its rollout; the environment and RNG state are restored
after warm-up so those calls do not affect the episode or its measured cost.
Episode Latency is the mean per-episode sum of inference latency. Relative
Compute is the method's mean episode cost divided by Fixed-Full's mean episode
cost, reported as a percentage. When the hardware cost table is available,
episode cost is the per-configuration latency multiplied by the number of WAM
calls; otherwise, it uses measured local inference wall time summed over the
episode.

`core/sim_state_manager.py` provides `SimStateManager.save_state(env)` and
`SimStateManager.restore_state(env, snapshot)` for replayable LIBERO
counterfactual branches. Snapshots include the MuJoCo/robosuite engine state,
mutable scene transforms, global and environment RNG state, robot controller
state, environment counters, and robosuite observable sampling/cache state.
Validate exact state and dual-camera replay determinism with:

```bash
./.conda/fastwam/bin/python -m unittest tests.test_determinism -v
```

Check EGL rendering of both LIBERO cameras without loading the policy:

```bash
python scripts/check_env.py \
  --task-suite libero_spatial --task-id 0 --seed 0 --resolution 224 \
  --output-dir outputs/phase1/env_smoke
```

Run one real base-checkpoint episode and save the two-camera MP4 plus episode
metrics:

```bash
python eval_single_rollout.py \
  --model_path checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --task-config libero_uncond_2cam224_1e-4 \
  --task-suite libero_spatial --task-id 0 --seed 0 \
  --H 0 --D 0 --W 0 --E 10 --max-steps 400 \
  --gpu-ids 0,1,2,3 \
  --output-dir outputs/phase1/baseline_rollout
```

Each command refuses to overwrite files in a non-empty output directory.
`check_env.py` saves per-camera PNGs, a side-by-side preview, and
`env_check.json`. `eval_single_rollout.py` saves `episode.mp4` and
`episode.json`; one successful episode is only a smoke test, not a benchmark
success-rate estimate. The EGL backend is set before MuJoCo is imported.

## Phase 2: counterfactual state oracle

`collect_counterfactual_data.py` scans seeded task episodes, saves each selected
simulator checkpoint plus Python/NumPy/PyTorch RNG state, then restores every
state in isolated configuration branches. Its `--adapter module:function`
factory supplies the simulator and policy integration; each adapter must
implement `start_task(task_id, seed)`, `capture_checkpoint()`,
`restore_checkpoint(checkpoint)`, `infer_actions(observation, config)`,
`step(action) -> StepResult`, `is_success(observation, info)`, and `close()`.
`infer_actions` returns an action chunk and the collector executes at most E
actions before replanning. Adapters should reset model-side recurrent state in
`start_task` and include any adapter-owned random-generator state in simulator
checkpoints. `capture_libero_checkpoint` and `restore_libero_checkpoint` wrap
the flattened MuJoCo state APIs used by this repository's pinned LIBERO env.

Example end-to-end smoke run (deterministic fake simulator; not a task result):

```bash
python collect_counterfactual_data.py \
  --adapter tests.fake_counterfactual_adapter:create_adapter \
  --task-suite fake --task-ids 0 --seeds 0 \
  --gpu-ids cpu --execution-horizons 1 --scan-stride 2 \
  --max-states-per-episode 2 --max-steps 8 \
  --output-dir outputs/counterfactual/smoke
python analyze_oracle.py \
  --results outputs/counterfactual/smoke/counterfactual_results.jsonl \
  --cost-table cost_table.json --budget-ms 10 \
  --output-dir results/oracle/smoke
```

### Real FastWAM/LIBERO one-command run

With a compatible FastWAM IDM checkpoint and the required FastWAM/LIBERO
assets installed, launch all three stages with:

```bash
python run_phase2_pipeline.py --model_path <PATH> --num_gpus 10
```

The pipeline profiles the 37 H/D/W configurations in FP16, scans LIBERO
counterfactual branches, then writes the Oracle JSON/CSV reports. Each real
FastWAM replica currently requires four GPUs: `--num_gpus 10` therefore starts
two parallel model workers (8 GPUs total); the remaining two GPUs are unused.
The state scan itself runs on the first four-GPU group. Use `--dry-run` to print
the planned commands without starting GPU work. Existing non-empty run
directories are not overwritten; choose a unique `--run-name`.

For fake or custom adapters, `--gpu-ids 0,1,...` can still start one spawned
worker per listed GPU; real FastWAM uses four-GPU worker groups. The collector
refuses to overwrite an existing collection directory.
Each return Q is the discounted reward from the saved state to termination or
the step limit. `analyze_oracle.py` writes a global/task/state selection report,
a comparison CSV, and per-state choices. Its compute cost is Phase 1 latency
for H/D/W multiplied by observed inference calls (therefore E changes cost
through call count). The current Phase 1 cost table measures the small profiling
stub, not a pretrained FastWAM checkpoint; use the one-command pipeline (or
`profile_cost_table.py --model_path <PATH>`) to regenerate the table with the
actual policy before interpreting budget or utility numbers as real-model
results. H=0 entries normalize D/W to zero in the grid; the real adapter uses
`--direct-depth` denoising steps for its FastWAM direct-action branch, and
records that value in the cost-table metadata.
Oracle selections are empirical in-sample upper bounds; use held-out episodes
for generalization claims.

### Fixed-baseline replay snapshots

`oracle/collect_snapshots.py` pairs No-Imagination failures with Fixed-Full
successes, captures five states before each watershed failure, and replays
successful episodes under Fixed-Full. Each physical stage can contribute up to
two separated states per episode. `--snapshots-per-category` and
`--minimum-per-category` control target and warning thresholds (defaults 50 and
40). The collector reports when the CSV cannot supply independent episodes for
the target quota rather than duplicating states. The current Phase 1 CSV has 15
unique episodes, so it cannot produce 40-60 samples in every category; additional
task/seed baselines are required. Stage labels use gripper/object distance,
object displacement, gripper contact force, and BDDL goal distance.

LIBERO's configured `datasets` directory is resolved through a temporary config:
`--trajectories-dir` selects a recorded-data root, otherwise the collector
searches `outputs/` for `.hdf5`, `.h5`, and `.pkl` files, then falls back to the
configured empty `no_demonstrations` placeholder. The episode CSV contains
summary fields, not trajectory paths; absent recorded files, rollouts are
regenerated from each CSV row's `task_suite`, `task_id`, and `seed`.

```bash
python -m oracle.collect_snapshots \
  --episodes-csv results/fixed_baselines_episodes.csv \
  --output-dir snapshots/physical_pool \
  --snapshots-per-category 50 --minimum-per-category 40 \
  --max-per-episode-per-phase 2 --phase-sample-gap 5 \
  --trajectories-dir outputs/baseline/EXP-B000/no_demonstrations \
  --model-path checkpoints/fastwam_release/libero_optional_idm_2cam224.pt
python oracle/counterfactual_oracle.py \
  --snapshots-dir snapshots \
  --cost-table configs/hardware_cost_table.json \
  --output-csv results/oracle_counterfactual_results.csv \
  --eval-mode sustained --num-repeats 3 \
  --gpu-ids 0,1,2,3 \
  --model-path checkpoints/fastwam_release/libero_optional_idm_2cam224.pt
python -m oracle.analyze_oracle \
  --results-csv results/oracle_counterfactual_results.csv \
  --cost-table configs/hardware_cost_table.json --output-dir results
```

The Oracle evaluates `B_action`, `B_small`, `B_deep`, `B_wide`, `B_far`, and
`B_full`; `--eval-mode sustained` keeps the selected configuration active until
the episode ends and records actual branch time, binary success, sparse reward,
dense goal-distance reward, and cumulative relative compute. Analysis uses
`Q = 0.8 * success + 0.2 * normalized_dense_reward` and writes the RQ1/RQ2
plots and reports under `results/`.

The experiment protocol is recorded in `EXPERIMENT_LOG.md`.

## Experiment progress and results

The current plan stage, acceptance checklist, execution history, and measured
results are tracked in [`docs/experiment_tracking/`](docs/experiment_tracking/README.md).
Start with its [current-stage status](docs/experiment_tracking/current_stage.md);
the [results summary](docs/experiment_tracking/experiment_results.md) links to
the real FP16 37-configuration cost table and states the limits of the current
evidence.
