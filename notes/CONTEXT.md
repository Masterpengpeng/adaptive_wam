# Research Context

> Updated: 2026-09-13 12:11 +08:00  
> Evidence through: Git inspection at `6c805bf` and `experiment_runner` read-only report  
> Canonical record: [`../EXPERIMENT_LOG.md`](../EXPERIMENT_LOG.md)

This file is a replaceable current-state snapshot for quickly resuming work. Formal
experiment results and complete handoffs belong in `EXPERIMENT_LOG.md` after review
by the main agent.

## Now

- Stage: baseline acquisition and code reading
- Active experiment: none (`EXP-B000` has not started)
- Objective: identify and pin the official baseline before environment setup or evaluation
- Status: ready for read-only baseline inspection
- Owner: main agent / `experiment_runner`

## Fixed / Do Not Change

- Baseline candidate: <https://github.com/yuantianyuan01/FastWAM>
- Baseline commit: TBD; record the exact SHA immediately after cloning
- Checkpoint: TBD
- Benchmark: LIBERO; tasks and seeds TBD
- `H / D / W / E`: TBD; do not change more than one axis in an experiment
- Do not interpret `action_horizon` as `H`; verify its runtime semantics and code path first
- Do not use or repair the base Python environment for this project
- Do not overwrite outputs; before long runs, report GPU count, exact command, and output path
- Estimate repository, checkpoint, dataset, and output disk usage before downloading artifacts

## Last Verified State

- Project HEAD: `6c805bf870d67e59f37d3e17ef74af89f5d5581c` (`Record EXP-000 initialization results`)
- Worktree: clean before this context file was created
- `repos/`: empty
- Environment: dedicated Conda environment not created; base Python has an incomplete
  user-level `torch 2.13.0` installation and is unsuitable for the baseline
- Hardware: 10 × RTX 2080 Ti (11,264 MiB each); CUDA toolkit 12.8
- Storage: approximately 55 GiB free on the workspace filesystem at last inspection
- Latest experiment command: none
- Latest experiment output: none
- Result: project initialization (`EXP-000`) is complete; no training or evaluation has run

## Open Issues

- [ ] Confirm that the FastWAM candidate is the official repository and pin its commit SHA.
- [ ] Confirm checkpoint location, size, checksum/version, and license/access requirements.
- [ ] Locate the LIBERO evaluation entry point and determine the real code mappings of
      `H / D / W / E` without guessing from parameter names.
- [ ] Create an isolated environment only after official dependency requirements are understood.
- [ ] Recheck free space and prepare a download budget before fetching checkpoints or data.

## Decisions Since Last Handoff

- Clone the candidate repository without installing dependencies, then pin its SHA and inspect
  the official entry points.
- Treat `action_horizon` as semantically unverified; it must not be recorded as `H` based only
  on its name.
- Keep following `DEC-001` through `DEC-003`: Oracle before Controller, one axis at a time,
  and public-checkpoint inference/evaluation before large-model training.

## Next Exact Action

1. Check the current disk budget, then clone only
   <https://github.com/yuantianyuan01/FastWAM> into `repos/` without installing dependencies.
2. Record the cloned repository's exact commit SHA.
3. Read `README.md`, `pyproject.toml`,
   `experiments/libero/run_libero_manager.py`,
   `experiments/libero/eval_libero_single.py`, and `configs/sim_libero.yaml`.
4. Report repository provenance, dependency/environment requirements, checkpoint procedure,
   LIBERO entry point, and evidence-backed `H / D / W / E` mappings before making code changes.

## Recent Evidence

- `2026-09-13 12:11 +08:00` — local Git inspection — branch `main`, clean worktree,
  HEAD `6c805bf870d67e59f37d3e17ef74af89f5d5581c` before this file was added.
- `2026-09-13` — `experiment_runner` read-only report — `repos/` is empty; FastWAM is the
  official-repository candidate; next operation is clone-only inspection with no installation.
- `2026-09-13` — `notes/environment_2026-09-13.md` — base PyTorch import fails and the
  workspace filesystem had approximately 55 GiB free.
