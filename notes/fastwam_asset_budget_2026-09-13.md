# FastWAM Asset Budget — 2026-09-13

This audit used repository metadata APIs only. No checkpoint or model asset was downloaded.

## Revisions

- FastWAM Hugging Face repository: `yuanty/fastwam@8eaceeb24c3cc92ff2a9c9a9d266a4941b836705`
- Converted Wan assets on ModelScope: `DiffSynth-Studio/Wan-Series-Converted-Safetensors`, observed `master` head `150f75d811d51f6c7760154aa7fec371dccda529`
- Tokenizer source on ModelScope: `Wan-AI/Wan2.1-T2V-1.3B`, observed `master` head `020b9e59db399efa534e408fb295b14a75700daf`

Metadata sources:

- <https://huggingface.co/api/models/yuanty/fastwam?blobs=true>
- <https://www.modelscope.cn/api/v1/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/repo/files?Revision=master&Root=>
- <https://www.modelscope.cn/api/v1/models/Wan-AI/Wan2.1-T2V-1.3B/repo/files?Revision=master&Root=google%2Fumt5-xxl>

## Required assets

| Asset | Bytes | GiB | SHA-256 |
|---|---:|---:|---|
| FastWAM base checkpoint | 12,041,735,140 | 11.213 | `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579` |
| Base dataset statistics | 40,939 | <0.001 | Git blob `e8ff7341ad86b9750dc1fbc26250da8c2c05f1a2` |
| Optional-IDM checkpoint | 12,041,735,545 | 11.213 | `26b4efffded221b9a303ca8f2cddb9999c8b7524e2751c855c74a986242ce8b4` |
| Optional-IDM statistics | 40,939 | <0.001 | Same Git blob as base statistics |
| UMT5-XXL BF16 encoder | 11,361,845,432 | 10.582 | `d92de679881d38af9c89eff7bb1b6d6c9d96cb2b69831e4027e9ecabdd38eb23` |
| Wan2.2 VAE | 1,409,401,152 | 1.313 | `0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996` |
| UMT5 tokenizer files | 21,454,081 | 0.020 | See source metadata per file |

## Totals

- Base-only minimum model assets: `24,834,476,744` bytes = `23.129 GiB`
- Optional-IDM-only minimum model assets: `24,834,477,149` bytes = `23.129 GiB`
- Base and Optional-IDM together with shared assets: `36,876,253,228` bytes = `34.344 GiB`
- Free workspace capacity at audit: approximately `54.58 GiB`

Estimated base-only safe peak budget:

| Category | Budget |
|---|---:|
| Exact model assets | 23.13 GiB |
| Isolated environment and CUDA-enabled dependencies | 8 GiB |
| Temporary package/download caches | 5 GiB |
| LIBERO, MuJoCo, and task assets without demonstrations | 2 GiB |
| Compile/runtime cache | 3 GiB |
| Smoke-test outputs | 2 GiB |
| Filesystem safety reserve | 8 GiB |
| Total | approximately 51.1 GiB |

## Decision

- Base only: conditional disk GO, with only approximately 3.4 GiB beyond the safe peak budget.
- Base plus Optional-IDM simultaneously: disk NO-GO; estimated safe peak is approximately 62.3 GiB.
- Actual evaluation on one RTX 2080 Ti: NO-GO under the official loading path. The 11.21 GiB checkpoint, 10.58 GiB text encoder, 1.31 GiB VAE, and activations cannot reasonably fit into 11 GiB VRAM.

Do not download weights until a verified offload/sharding plan or a larger-memory GPU is available. The next safe step is an isolated environment plus a no-weight CUDA/BF16 compatibility smoke test.
