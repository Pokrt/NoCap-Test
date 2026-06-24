# IDEA.md — Jan Kadlec (kadlej27@student.cvut.cz)

**TL;DR:** I tried three efficiency ideas (sliding-window attention, log-frequency
bias init, forced FlashAttention). None beat the baseline. The real finding is
methodological: I ran on a shared HPC cluster (MetaCentrum) where the scheduler
assigns a random GPU, and the *identical* training job varies **4.8× in wall-clock**
(2.4 h on an H100 → 11.5 h on an A40). That hardware variance dwarfs any
algorithmic speedup, so I built a dashboard to filter W&B runs by GPU model and
only then compared variants on matched hardware. All numbers below are computed
from 43 logged runs. Full write-up: [`report/paper.pdf`](report/paper.pdf).

## Setup

Baseline GPT-2 small (`d12`, 124M), 4768 steps, eff. batch 512×1024 (~2.5B tokens),
target val loss **3.3821** on FineWeb. Hyperparameters left at baseline values
throughout — I only change the algorithmic factor under test, and I compare each
variant to the baseline **on the same GPU model**.

## The problem I actually hit: hardware variance

Identical baseline job, by GPU (W&B process wall-clock):

| GPU | n | wall-clock | val loss |
|---|---|---|---|
| H100 NVL | 1 | 2.38 h | 3.378 |
| RTX PRO 6000 Blackwell | 1 | 2.62 h | 3.381 |
| A100-SXM4-40GB | 3 | 3.93 h | 3.379 |
| L40S | 12 | 5.24 h | 3.381 |
| A40 | 3 | 9.33 h | 3.380 |

Slowest/fastest single run = **4.84×**. Even two A40 runs of the same job differ
by ~40% (29.5k vs 41.5k s). Validation loss is invariant to the GPU
(3.380 ± 0.0015, sd = 0.00146 over 20 baseline runs) — only time changes.
**Takeaway:** a 5–10% algorithmic speedup (~0.5 h) is unmeasurable against a
±4 h scheduling effect unless you hold the GPU fixed. Hence the dashboard
(<https://gpt-dashboard.jan-kadlec.cz>, source <https://github.com/Pokrt/wandb-dashboard>):
W&B records the GPU in run metadata but doesn't expose it as a filter; the
dashboard makes "GPU model" a first-class filter so A/B comparisons are fair.

## What I tried

### 1. Sliding-window attention — ❌ worse loss *and* slower
Restrict each token to the previous `w` tokens (Longformer/Mistral style),
`w ∈ {64, 128, 256}`. Implemented as an additive band+causal `attn_mask` to SDPA.

Same-GPU (L40S) result:

| variant | wall-clock | val loss |
|---|---|---|
| full attention | 5.24 h | 3.380 |
| window 256 | 5.93 h | 3.399 |
| window 128 | 6.00 h | 3.441 |
| window 64 | 5.79 h | 3.498 |

Monotonic degradation, far outside the noise floor — a 124M model at T=1024 needs
global context to reach target. And it was **slower**: passing an explicit float
`attn_mask` disables SDPA's fused causal fast path and falls back to a slower
kernel. Double loss.

### 2. Forced FlashAttention — ❌ no-op (the baseline already runs flash)
Wrapped SDPA in `sdp_kernel(enable_flash=True, ...)`. On matched A100s: 3.92 h vs
3.93 h baseline (−0.2%), loss within noise. Reason: **the baseline already runs
FlashAttention.** PyTorch's `F.scaled_dot_product_attention` is a *dispatcher*
that auto-selects a fused flash kernel for causal attention on Ampere-and-newer
hardware, and every GPU in my pool (A40, L40S, A100, H100, RTX PRO 6000) is
Ampere+. I did not initially know SDPA defaults to flash — so "adding flash
attention" was really just re-enabling a kernel that was already on. Classic
phantom optimisation.

Two dead ends underneath this: (a) the third-party `flash-attn` package — ~1 h
source build, ~10 GB footprint — kept blowing the cluster scratch quota; (b) an
earlier code path auto-enabled that package on successful import (no flag), and 4
runs were launched with it, but since it never installed, the `try/except`
silently fell back to SDPA. W&B's captured `pip freeze` confirms `flash_attn` was
absent in all 4, and their times match the SDPA baselines on the same GPU. **No
run in the project ever executed the standalone library — everything used SDPA's
flash kernel, baseline included.**

### 3. Log-frequency bias init — ➖ marginal, within noise
Init the (newly added) output-layer bias to `log f_i`, the log unigram frequencies
of the training data with Laplace smoothing (ε=1), so the model starts at the
unigram distribution (initial loss ≈ corpus entropy instead of `log V ≈ 10.8`) and
only learns deviations.

Same-GPU (A40): 3.379 vs 3.380 baseline, Δ = −0.0015 (≈ 1σ). The single best val
loss across all 43 runs (**3.3758**) is a log-freq run, and the direction is
consistent — but it's at the noise floor, so I can't claim significance. Free at
runtime and well-motivated, so a reasonable default; not a speedup.

## What didn't work / lessons

- The "obvious" efficiency wins were redundant (flash) or harmful (smaller window).
- Most effort went to infrastructure, not modelling: a CLI `qsub -l select=…`
  silently overriding the in-script directive and dropping `scratch_local` (→
  out-of-disk), Kerberos blocking non-interactive submission, and unstable relative
  paths on compute nodes.
- **Recommendation:** never report a speedrun result without a same-hardware
  baseline; pin the node (`vnode=`) or run baseline+variant as a pair in one
  allocation; always log the GPU and make it filterable.

## Reproduce
Branches: `sliding-window-attention`, `log-freq-bias-init`, `flash-attention`.
Flags: `--window_size`, `--log_freq_init`, `--flash_attn`. Runs in W&B project
`honza-kadlec-ctu-fee/benchmark_gpt2`.
