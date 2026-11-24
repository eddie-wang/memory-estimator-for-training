# Memory Estimator for Training

A single-file, interactive HTML calculator for the **per-GPU memory** and the **minimum GPU count** needed to train large (MoE or dense) language models with **FSDP × context/sequence parallelism (CP/SP)** and optional **expert parallelism (EP)**.

Open `index.html` in a browser; there's no build step and no dependencies. You can also serve it with GitHub Pages.

## What it models

N = X (FSDP) × Y (CP/SP) GPUs. Model state is sharded over all N GPUs, and each sequence is split across the Y GPUs of its CP group.

```
tokens per GPU      T  = S × MBS / Y
model states       = (4 + b_opt) × P / N                 b_opt = 12 (Adam) or 8 (Muon)
layer checkpoints  = L × 2 B × H × T                     full activation checkpointing
transient          = w × T × H × 2 B                     one layer's recompute, MoE kernels, attention, backward
logits             = n_logit × T × V × 4 B               ~0 at peak with a chunked loss
FSDP gather        = c × 2 B × [ (P − E)/L + E/(L × EP) ]   c live copies of one layer's local bf16 params
allocated peak     = states + checkpoints + transient + logits + gather
reserved peak      = allocated × (1 + frag)
fits if            reserved peak ≤ GPU capacity − non-PyTorch reserve
```

- Only **model states** shrink as you add FSDP GPUs. Checkpoints, transient memory and logits depend only on tokens per GPU (S / Y), so CP/SP is the lever for them.
- **EP** only changes the FSDP gather term: each GPU gathers just its own 1/EP share of the routed experts. Non-expert weights are not divided. Model states stay at 1/N, and token-dispatch buffers (~T × top-k × H_moe × 2 B) don't depend on EP. Higher EP can increase routing imbalance on the worst rank.
- Persistent state per parameter (measured): fp32 weights (4) + fp32 grads (4) + optimizer state, which is fp32 momentum (4) for Muon or m and v (8) for Adam. That's 12 B/param for Muon and 16 B/param for Adam.

## Calibration

The per-token terms are calibrated from a PyTorch allocator snapshot (`torch.cuda.memory._dump_snapshot`) of a GLM-5.2-scale MoE run. The run had 744B params on 64 × B300, FSDP 16 × SP 4, EP 8, Muon, and packed sequences up to 131K.

| Component at peak | GiB |
|---|---|
| Parameter shards (fp32) | 43.2 |
| Optimizer momentum (fp32) | 43.2 |
| Gradient shards (fp32) | 43.3 |
| Layer-input checkpoints | 19.9 |
| MoE kernel overhead (dispatch, permute/scatter, all-to-all, grouped-GEMM outputs) | ~14 |
| Sparse attention kernels, SP all-to-all, indexer | ~13 |
| FSDP all-gather / reduce buffers | ~5 |
| Other transient | ~9 |
| **Total** | **190.4** |

Default knobs:

- Transient multiplier w: 136 for that snapshot, or 190 as a pessimistic value from a second run's peak at the same topology.
- Logits: n_logit = 0.
- FSDP gather: c = 2 (measured 1.9).
- Non-PyTorch reserve: 8 GiB.
- Fragmentation: 0%.

## Presets

GLM-5.2/5.3, Kimi K3/K3.1, DeepSeek-V3, Qwen3-235B-A22B, Llama-3 70B/8B. Every value can be edited.

## Caveats

- The per-token terms were calibrated on one MoE + sparse-attention model family. Dense models, other attention kernels, or other MoE kernel implementations will have a different `w`, so recalibrate from your own allocator snapshot.
- Memory is assumed to scale linearly in tokens per GPU, and the worst rank (longest packed batch, most uneven expert load) sets the real peak.
- Throughput, batch-size divisibility and communication cost are not modeled. Fitting in memory is necessary but not sufficient.
