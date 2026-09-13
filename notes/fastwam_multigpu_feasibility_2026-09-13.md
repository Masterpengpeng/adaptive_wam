# FastWAM Multi-GPU Feasibility Audit — 2026-09-13

This was a static source audit against FastWAM commit `7faa71108368fbb3b6885649f112af607427a2d4`. No model weight was downloaded or loaded, and no baseline code was changed.

## Finding

There is no configuration-only solution for an 11 GiB RTX 2080 Ti. The official path places the text encoder, VAE, video expert, and action expert on one device and performs a final whole-model `.to(device)`.

Static weight lower bounds:

- MoT: 6,020,710,599 parameters, approximately 11.214 GiB in BF16
- UMT5-XXL: approximately 10.582 GiB in BF16
- VAE weights: approximately 1.313 GiB
- Approximate simultaneously resident weights in the official path: 23.11 GiB before CUDA context, activations, KV cache, and workspace

The MoT weights alone exceed the 10.568 GiB device capacity reported by PyTorch for each RTX 2080 Ti.

## Semantics-preserving candidate

For the published base/Optional-IDM first-frame path:

1. Disable `compile_action_infer` during correctness validation.
2. Encode each task prompt once, cache its context, then offload T5. T5 itself requires two 2080 Ti devices for CUDA encoding; CPU BF16 is a much slower fallback and may not be bitwise equivalent.
3. Keep the video expert on GPU 0.
4. Keep the action expert and VAE on GPU 1, or use a separate VAE device if measured peaks require it.
5. Encode the current frame with the VAE and move the small latent to the video device.
6. Run video-expert KV prefill and transfer the approximately 34 MiB first-frame KV cache to the action device.
7. Run action denoising on the action device.

This preserves the model operations, weights, scheduler, mask, and seed, but changes device placement and introduces cross-device transfers.

For IDM, the same structure can transfer an approximately 103 MiB full-video KV cache after the video stage. Joint inference cannot simply put one expert on each GPU because each layer concatenates video/action QKV on the same device.

## Fallbacks

- Paired-layer partitioning: colocate corresponding video/action blocks and split contiguous layer ranges across two devices. More invasive, but supports Base, Joint, and IDM with greater per-device headroom.
- Sequential single-device offload: mathematically viable but likely too slow because each replan can move roughly 20–25 GiB of weights over PCIe, and the 9.313 GiB video expert leaves little activation workspace.
- Accelerate `device_map`: does not automatically handle this custom MoT cross-expert computation or the final blanket `.to()`.
- FSDP: possible in principle, but multi-process full-shard inference and per-layer all-gather are disproportionate and poor fits for batch-one latency evaluation.

## Required patch boundary

- Add explicit component devices during loader/runtime construction.
- Construct and load on CPU before placing individual components.
- Remove blanket whole-model `.to(device)` calls.
- Separate text, VAE, video, and action device ownership.
- Cache prompt context at task scope.
- Move VAE latents and KV cache explicitly to their consumer devices.
- Create scheduler tensors, masks, and static conditioning on the correct consumer device.

No scheduler, seed, attention rule, model weight, or inference equation should change.

## Validation gate

Compare against a sufficient-memory reference using identical checkpoint, input, prompt, seed, and D:

- T5 context
- VAE latent
- every layer's KV cache
- each denoising-step latent
- final action

Target maximum error is `1e-3`; minimum acceptance is the repository's existing action-parity tolerance of `atol=rtol=1e-2`. Run 1, 10, and 100 replans while recording each GPU's peak allocated/reserved memory and host RSS, confirming no monotonic leak.

## Decision required

- Preferred scientific path: run the untouched official baseline on a GPU with at least 24–32 GiB VRAM.
- Local fallback: authorize a separate branch containing the multi-GPU device-placement patch and treat numerical parity as a prerequisite before reporting baseline results.

