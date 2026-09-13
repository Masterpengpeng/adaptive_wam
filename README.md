# Adaptive Inference for World Action Models

This repository is the experiment workspace for studying state-adaptive inference configurations in World Action Models.

The four inference variables are:

- `H`: future prediction / imagination horizon
- `D`: denoising / refinement depth
- `W`: number of stochastic future rollouts
- `E`: execution horizon / action-chunk commitment length

## Current stage

`EXP-000` project initialization and environment validation.

No baseline algorithm has been modified and no long-running evaluation has been started.

## Directory layout

- `repos/`: third-party baseline repositories
- `configs/`: versioned experiment configurations
- `scripts/`: evaluation, profiling, and oracle tooling
- `checkpoints/`, `data/`: local large artifacts (ignored by Git)
- `outputs/`, `logs/`: raw experiment artifacts (ignored by Git)
- `results/`: processed tables and figures
- `notes/`: environment and research notes

The experiment protocol is recorded in `EXPERIMENT_LOG.md`.

