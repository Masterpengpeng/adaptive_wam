# Environment Snapshot — 2026-09-13

Captured at `2026-09-13T12:02:08+08:00` before baseline setup.

## Host

- Hostname: `lab416-server`
- Memory: 157 GiB total, 152 GiB available
- Workspace filesystem: 468 GiB total, 55 GiB available (88% used)

## GPU and CUDA

- GPU: 10 × NVIDIA GeForce RTX 2080 Ti
- Memory per GPU: 11,264 MiB
- Driver: 570.211.01
- GPU state at inspection: all GPUs at 0% utilization and 6 MiB memory used
- CUDA compiler: 12.8 (`nvcc V12.8.61`)
- Note: `nvidia-smi` cannot access the driver inside the restricted command sandbox, but succeeds in the host environment.

## Toolchain

- Python: 3.12.7 (`/home/lab416/anaconda3/bin/python`)
- Conda: 24.9.2; active environment: `base`
- GCC: 11.4.0
- Git: 2.34.1

## PyTorch status

The active `base` environment is not suitable for the baseline:

- `pip` reports `torch 2.13.0` under the user site-packages directory.
- Import fails with `ModuleNotFoundError: torch.fx.experimental.unification.core`.
- Disabling user site-packages leaves no PyTorch installation in the Conda base environment.

Do not repair or reuse this environment for the experiment. Create a dedicated Conda environment from the selected baseline's pinned official requirements.

## Readiness decision

Hardware and system toolchain are available. Baseline repository inspection/clone may proceed, followed by creation of an isolated environment. Full evaluation must wait until that environment passes a PyTorch/CUDA smoke test.

