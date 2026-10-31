# Run the decoupled experiment on another server

The repository contains the 20 curated snapshots, the ten-configuration grid,
the analysis code, and a patch that reproduces the local four-GPU FastWAM
implementation. Large model assets are downloaded from immutable Hugging Face
revisions and verified before use.

## Requirements

- Linux with Conda and Git
- a compatible NVIDIA driver for PyTorch 2.7.1 / CUDA 12.8
- four distinct CUDA GPUs with at least 11 GiB each
- at least 45 GiB of free disk space

## Setup

```bash
git clone git@github.com:Masterpengpeng/adaptive_wam.git
cd adaptive_wam
bash scripts/bootstrap_decoupled_server.sh --download-assets
```

The download is resumable and installs approximately 26.24 GB of verified
Optional-IDM, UMT5, VAE, and tokenizer assets. To transfer those directories
from an existing server instead, omit `--download-assets` and place the files
at the paths recorded in `results/manifest/decoupled_assets.json`.

Bootstrap also reconstructs the exact local FastWAM multi-GPU changes from the
checked-in patch and generates `.runtime/libero/config.yaml` for the current
checkout. No path from the original server is reused.

## Run

```bash
bash scripts/run_decoupled_experiment.sh 0,1,2,3
```

The command resumes an existing CSV, targets 200 rows, and then writes the
single-axis analysis under `results/oracle_decoupled_v1/`.

The checked-in hardware cost table is intentionally retained for comparison
with the original RTX 2080 Ti experiment. Re-profile it separately when the
research question requires costs measured on the new hardware.
