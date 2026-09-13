# FastWAM Core CUDA Environment Smoke Test — 2026-09-13

## Scope

Create an isolated Python/PyTorch environment and verify CUDA, FP16, and BF16 execution without installing FastWAM/LIBERO or downloading model assets.

## Environment

- Prefix: `/home/lab416/ZP_projects/adaptive_wam/.conda/fastwam`
- Python: 3.10.21
- PyTorch: 2.7.1+cu128
- torchvision: 0.22.1+cu128
- PyTorch CUDA runtime: 12.8
- NumPy: 2.2.6
- Triton: 3.3.1

The following environment variables are stored in the prefix:

```text
PYTHONNOUSERSITE=1
PIP_CACHE_DIR=/home/lab416/ZP_projects/adaptive_wam/.conda/pip-cache
XDG_CACHE_HOME=/home/lab416/ZP_projects/adaptive_wam/.conda/xdg-cache
CONDA_PKGS_DIRS=/home/lab416/ZP_projects/adaptive_wam/.conda/pkgs
```

This prevents the broken user-level Python packages and global caches from contaminating the experiment.

## Installation commands

Equivalent reproducible commands:

```bash
CONDA_PKGS_DIRS=/home/lab416/ZP_projects/adaptive_wam/.conda/pkgs \
XDG_CACHE_HOME=/home/lab416/ZP_projects/adaptive_wam/.conda/xdg-cache \
conda create -p /home/lab416/ZP_projects/adaptive_wam/.conda/fastwam python=3.10 pip -y

PYTHONNOUSERSITE=1 \
PIP_CACHE_DIR=/home/lab416/ZP_projects/adaptive_wam/.conda/pip-cache \
/home/lab416/ZP_projects/adaptive_wam/.conda/fastwam/bin/pip install \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

`pip check` reports `No broken requirements found` with user site-packages disabled.

## GPU results

- Host CUDA availability: true
- Device count: 10
- Device 0–9: NVIDIA GeForce RTX 2080 Ti
- Compute capability: 7.5
- Total memory reported by PyTorch: 11,348,672,512 bytes per GPU
- `torch.cuda.is_bf16_supported()`: true on every GPU

Each GPU ran synchronized 64×64 matrix multiplication in FP16 and BF16:

- Tests: 20 total (10 GPUs × 2 dtypes)
- Finite outputs: 20/20
- Failures: 0
- Peak allocated per device/dtype: 8,564,736 bytes
- Peak reserved per device/dtype: 23,068,672 bytes

BF16 execution passing does not imply native BF16 Tensor Core throughput on Turing. Performance must be measured independently and not inferred from newer GPU architectures.

## Disk impact

- Free before: 58,603,728,896 bytes = 54.58 GiB
- Free after: 50,016,043,008 bytes = 46.58 GiB
- Net consumption: approximately 8.00 GiB
- `.conda` total: approximately 7.93 GiB
- Environment prefix: approximately 6.49 GiB
- pip cache: approximately 1.16 GiB
- Conda package cache: approximately 393 MiB

## Decision

- Core Python/PyTorch/CUDA environment: GO
- FP16/BF16 functional smoke test: GO
- FastWAM/LIBERO dependency installation: not yet performed
- Model evaluation: still NO-GO under the official one-device loading path because the checkpoint, text encoder, VAE, and activations exceed 11 GiB VRAM

No model asset, checkpoint, or LIBERO dataset was downloaded.

