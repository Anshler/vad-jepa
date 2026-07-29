# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is **MOVAD + V-JEPA 2.1**: a supervised anomaly detection pipeline for traffic dashcam video. It replaces MOVAD's original Video Swin Transformer backbone with a **frozen V-JEPA 2.1 ViT encoder** and keeps the original LSTM temporal model + binary classifier. Multiple temporal model variants (LSTM, Mamba, SlotSSM, Sparse SlotSSM) are implemented as a controlled experiment — change one variable at a time and measure the impact on frame-level anomaly detection.

- **Dataset**: DoTA — dashcam traffic videos, 10 anomaly classes, ego-involvement labels, day/night splits. 10 FPS, 640×480 original resolution, resized to 256×256.
- **Input**: RGB frames only (no optical flow, no object detectors).
- **Output**: frame-level anomaly probability (softmax class 1 from binary classifier).
- **Pretrained encoder**: V-JEPA 2.1 (self-supervised, masked feature prediction on K710+SSv2+HowTo100M+ImageNet1K).

## Architecture and data flow

```
DoTA video (384x384, 10 FPS)
    ↓ buffer NF-frame sliding-window clips (stride-1)
V-JEPA 2.1 ViT encoder (frozen, no grad)
    ↓
    ├─ Standard path (LSTM / Mamba): spatial mean pool → [B, embed_dim]
    │   → LayerNorm → Linear(embed_dim → dim_latent) → ReLU → Dropout
    │   → LSTM(3 layers) or Mamba2(3 blocks)
    │   → Linear → ReLU → Dropout → Linear(→ 2)
    │
    └─ SlotSSM path: full patch tokens → [B, N, embed_dim]
        → SlotSSM(K slots, per-slot Mamba2, cross+self-attention)
        → learned attention-pool over slots
        → LayerNorm → Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear(→ 2)
```

**Key architectural decisions:**
- Encoder is always frozen; only the temporal model + classifier are trained.
- LSTM hidden state is detached at each step (no BPTT through full video).
- SlotSSM uses full V-JEPA patch tokens (not mean-pooled) so cross-attention can induce slot specialization.
- Mamba models use `MambaCache` for streaming inference (stride-1 sliding window without re-scanning).
- Sparse SlotSSM freezes inactive slots bit-for-bit across all three sub-steps (cross-attn, Mamba, self-attn). Inactive slots serve as read-only memory.

## Key files

| File | Purpose |
|------|---------|
| `main.py` | Entry point: CLI parsing, training loop, testing loop |
| `model.py` | `ClsVJEPA` + all temporal model variants + `MambaCache` + `build_cls_vjepa` factory |
| `vjepa_encoder.py` | Frozen V-JEPA 2.1 encoder wrapper; handles the `src` package name collision during import, loads pretrained weights, exposes `(clip) → features` interface |
| `movad_core/dota.py` | DoTA dataset class, annotation loading, sub-batch sampling, data transforms |
| `movad_core/losses.py` | Weighted cross-entropy loss builder |
| `movad_core/optim.py` | SGD optimizer builder (no LR scheduler by default) |
| `movad_core/metrics.py` | Frame-level AUC, PR-AUC, F1, per-class, ego-involvement breakdown |
| `movad_core/utils.py` | Checkpoint save/load, result pickling, per-class result splitting |
| `movad_core/data_transform.py` | Frame padding, random vertical/horizontal flip transforms |

## Config system

Everything is driven by YAML configs in `cfgs/`. Configs are loaded via `easydict.EasyDict` in `main.py:parse_configs()`. CLI args (`--epochs`, `--phase`, etc.) are merged in. Models are built via `build_cls_vjepa(cfg)` which reads the config and instantiates the correct temporal model variant.

### Available configs

| Config | Temporal model | Trainable params | Notes |
|--------|---------------|-----------------|-------|
| `vjepa_v1.yaml` | LSTM | ~27.3M | Baseline — original MOVAD design with V-JEPA backbone |
| `vjepa_mamba.yaml` | Mamba2 | ~21.9M | 3 Mamba2 blocks, d=1024, expand=2 |
| `vjepa_slotssm.yaml` | SlotSSM (dense) | ~19.0M | K=32, D=512, 4 blocks, standard cross-attn |
| `vjepa_sparse_slotssm.yaml` | Sparse SlotSSM | ~19.0M | top_k=16, entropy reg + ε-random |

### Config keys for temporal models

Common: `temporal_model`, `dim_latent`, `dropout`, `rnn_cell_num`
Mamba-specific: `mamba_d_state`, `mamba_d_conv`, `mamba_expand`, `mamba_version` (always `"mamba2"`)
SlotSSM-specific: `num_slots`, `slot_dim`, `num_ssm_blocks`, `top_k`, `eps_random`
Sparse-only (training): `entropy_weight` (CLI-level, controls gate entropy penalty in training loop)
Encoder: `model_name` (vit_base/vit_large/vit_giant_xformers), `num_frames`, `img_size`, `checkpoint_path`, `compile`

### Mamba2 constraint

`(d_model * expand / headdim) % 8 == 0` where headdim=64. At d=1024 expand=2: 2048/64=32, 32%8=0 ✓. At d=512 expand=2: 1024/64=16, 16%8=0 ✓.

## Commands

**Always run via WSL** with the `vjepa2-312` conda environment. The project lives on the Windows filesystem (`/mnt/d/...`) but all execution happens in WSL.

```bash
wsl bash -l -c 'cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vjepa_movad && "$HOME/miniconda3/bin/conda" run -n vjepa2-312 python main.py --config cfgs/vjepa_v1.yaml --phase train --epochs 200'
```

For shorter inline commands, use the Bash tool directly (the harness routes it through WSL automatically when in a WSL directory).

### Training

Single-head (one temporal model):
```bash
wsl bash -l -c 'cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vjepa_movad && "$HOME/miniconda3/bin/conda" run -n vjepa2-312 python main.py --config cfgs/vjepa_mamba.yaml --phase train --epochs 200'
```

Multi-head (one encode → multiple temporal models trained in parallel):
```bash
wsl bash -l -c 'cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vjepa_movad && "$HOME/miniconda3/bin/conda" run -n vjepa2-312 python main.py --config cfgs/vjepa_mamba.yaml cfgs/vjepa_slotssm.yaml cfgs/vjepa_sparse_slotssm.yaml --phase train --epochs 200'
```

Validation runs automatically every 10 epochs during training (default). Metrics (AUC, PR-AUC, F1, F1-mean, accuracy) are logged to TensorBoard under `val/<metric>` for each head. Results pickles are saved to `{head_output}/eval/results-{epoch:02d}.pkl`.

Disable validation:
```bash
wsl bash -l -c 'cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vjepa_movad && "$HOME/miniconda3/bin/conda" run -n vjepa2-312 python main.py --config cfgs/vjepa_mamba.yaml --phase train --no-enable_validation'
```

Validate every 5 epochs:
```bash
wsl bash -l -c 'cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vjepa_movad && "$HOME/miniconda3/bin/conda" run -n vjepa2-312 python main.py --config cfgs/vjepa_mamba.yaml --phase train --validation_epoch_step 5'
```

Resume from checkpoint:
```bash
wsl bash -l -c 'cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vjepa_movad && "$HOME/miniconda3/bin/conda" run -n vjepa2-312 python main.py --config cfgs/vjepa_mamba.yaml --phase train --epoch 50'
```

### Precomputed validation embeddings

One-time precompute to cache frozen encoder outputs for the val set — skips the ViT during validation, ~10× faster eval.

```bash
wsl bash -l -c 'cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vjepa_movad && "$HOME/miniconda3/bin/conda" run -n vjepa2-312 python scripts/precompute_val_embeddings.py --config cfgs/vjepa_mamba.yaml'
```

Verify 1 video before the full run:
```bash
wsl bash -l -c 'cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vjepa_movad && "$HOME/miniconda3/bin/conda" run -n vjepa2-312 python scripts/precompute_val_embeddings.py --config cfgs/vjepa_mamba.yaml --verify'
```

Saves to `{data_path}/embedding_val/{video_key}.pt`. Each file: `patches_full` (f16), `data_info` (f32), `v_len` (int). Filename is the metadata key — alignment is inherent. Detection in `main.py` is automatic: if `embedding_val/*.pt` exists, validation/test uses precomputed embeddings instead of raw video encoding.

### Testing / evaluation

```bash
wsl bash -l -c 'cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vjepa_movad && "$HOME/miniconda3/bin/conda" run -n vjepa2-312 python main.py --config cfgs/vjepa_mamba.yaml --phase test --epoch 190'
```

Multi-head evaluation (all heads evaluated in one pass, sharing encoder output):
```bash
wsl bash -l -c 'cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vjepa_movad && "$HOME/miniconda3/bin/conda" run -n vjepa2-312 python main.py --config cfgs/vjepa_mamba.yaml cfgs/vjepa_slotssm.yaml cfgs/vjepa_sparse_slotssm.yaml --phase test --epoch 190'
```

### Smoke tests (all 6 temporal variants + resolution flexibility)

```bash
wsl bash -l -c 'cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vjepa_movad && "$HOME/miniconda3/bin/conda" run -n vjepa2-312 python tests/test_inference.py'
wsl bash -l -c 'cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vjepa_movad && "$HOME/miniconda3/bin/conda" run -n vjepa2-312 python tests/test_inference.py --amp fp16'
```

### Latency benchmarking

```bash
wsl bash -l -c 'cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vjepa_movad && "$HOME/miniconda3/bin/conda" run -n vjepa2-312 python tests/benchmark_latency.py'
wsl bash -l -c 'cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vjepa_movad && "$HOME/miniconda3/bin/conda" run -n vjepa2-312 python tests/benchmark_latency.py --amp fp32'
```

### Encoder-only optimization benchmarks

```bash
wsl bash -l -c 'cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vjepa_movad && "$HOME/miniconda3/bin/conda" run -n vjepa2-312 python tests/bench_encoder_opts.py'
```

### Sparse Mamba correctness test

```bash
wsl bash -l -c 'cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vjepa_movad && "$HOME/miniconda3/bin/conda" run -n vjepa2-312 python tests/test_sparse_mamba.py'
```

## Dependencies

- **Required**: `torch`, `torchvision`, `pytorchvideo`, `easydict`, `scikit-learn`, `tqdm`, `tensorboard`, `pillow`, `numpy`, `pyyaml`
- **Optional but needed for Mamba/SlotSSM**: `mamba-ssm` + `causal-conv1d` (CUDA kernel packages from state-spaces/mamba)
- **Optional**: `flash-attn` (speeds up SlotSSM cross/self attention; must be built against your exact PyTorch+CUDA. Pre-built wheels at https://github.com/Dao-AILab/flash-attention/releases)

The code gracefully degrades: `_HAS_MAMBA_SSM` and `_HAS_FLASH_ATTN` flags control conditional imports.

## V-JEPA 2.1 encoder details

- The encoder loads from a pretrained checkpoint via `load_pretrained_encoder()` in `vjepa_encoder.py`.
- `checkpoint_path` in the YAML config points to the `.pt` file; `checkpoint_key` selects the state dict key (typically `"ema_encoder"`).
- Strict loading is attempted first; on failure, falls back to shape-matched loading with warnings.
- The `src` package name collision between the vjepa2 codebase and movad's `src/` directory is resolved by temporarily swapping `sys.modules` entries during import.
- The encoder is wrapped with `torch.compile(mode="default")` (config key `compile: false` to disable). `"reduce-overhead"` was swapped for `"default"` to avoid CUDA graph memory lock-in during eval mode.
- Encoder supports `return_patches=True` (returns `[B, N, embed_dim]` patch tokens for SlotSSM) or `return_patches=False` (spatial mean pool → `[B, embed_dim]`).
- Uses 3D Rotary Position Embeddings (RoPE) with `interpolate_rope=True` — supports variable frame counts and resolutions at inference.

## Training loop details

- Stride-1 sliding window: every frame gets an anomaly score. The first NF frames have no score.
- `loss.backward()` is called **per timestep** (not accumulated over the video).
- Weighted cross-entropy: normal=0.3, anomaly=0.7.
- State is carried across timesteps: LSTM tuples `(h, c)`, Mamba uses `MambaCache`.
- Sparse SlotSSM entropy penalty is added directly to the per-frame loss in the training loop (not inside the model).
- **Checkpoints**: `model.heads[name].state_dict()` is filtered to exclude the frozen encoder (saves ~600MB per ckpt per head). Only temporal + classifier weights are stored. `MultiHeadVJEPA.train()` overrides the default behavior — heads follow the requested mode but the encoder is always pinned to `eval()`.
- Checkpoints save at `cfg.snapshot_interval` epochs to `{cfg.output}/checkpoints/model-{epoch:02d}.pt`.
- **Validation during training**: runs every `validation_epoch_step` epochs (default 10, clamped to `--epochs` so it runs at least once). Uses the same `_evaluate_model()` code path as standalone testing — one encode for all heads, then each head evaluated independently. Metrics logged to TensorBoard under `val/auc_roc`, `val/auc_pr`, `val/f1`, `val/f1_mean`, `val/accuracy`.
- Evaluation results pickle to `{cfg.output}/eval/results-{epoch:02d}.pkl`.

## Evaluation metrics

Frame-level: AUC, PR-AUC, F1-score, F1-mean (balanced), accuracy.
Per-class breakdown: AUC, PR-AUC, F1-mean, accuracy for each of the 10 DoTA anomaly classes.
Ego-involvement breakdown: same metrics split by ego_involve=0 vs ego_involve=1.

## State management for streaming inference

All temporal models must support the `(x, state) → (output, new_state)` interface where `state=None` at the beginning of each video (or segment). The state is carried forward per-frame within a video and reset across videos.

- **LSTM**: state is `(h, c)` tuple; both detached after each step.
- **Mamba/Mamba2**: state is a `MambaCache` with `seqlen_offset` and `key_value_memory_dict` tracking per-layer conv/SSM states.
- **SlotSSM/Sparse SlotSSM**: state is a `MambaCache`; slot values themselves are read/written within the forward pass (not stored in state).

## Sparse SlotSSM gating

Only top-k slots update per timestep; inactive slots are frozen bit-for-bit across all sub-steps (cross-attn, Mamba, self-attn). Two mitigations for training stability:

1. **Entropy regularization** (`entropy_weight` in config, applied in training loop): penalizes peaked gate distributions to encourage uniform slot usage.
2. **ε-random activation** (`eps_random` in config, applied in `SlotSSMBlock.forward`): with probability ε, bypass the learned gate and select k slots uniformly at random.

Gate entropy, per-block, is logged to TensorBoard. The sparse path uses a compact-active implementation (integer-indexed dense tensors, no mask broadcasting) to keep CUDA graphs fused.

## Windows / WSL notes

- The primary development environment is WSL (conda env `vjepa2-312`).
- `pin_memory=False` on Windows (detected in `dota.py:setup_dota()`).
- Paths use forward slashes — the WSL `/mnt/d/...` convention.
- Tests and benchmarks assume running from WSL with the conda environment activated.
