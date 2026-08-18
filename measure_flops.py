"""Measure BOPs (or FLOPs) for the four generation modes using fvcore.

BOPs (bit operations) expose the theoretical compute reduction from lower
precision, unlike FLOPs.  This script uses the common convention
``BOPs = MACs * bits(left operand) * bits(right operand)``.  Linear and
convolution MACs use activation_bits * weight_bits.  The two attention
matmuls use activation_bits**2 and softmax_bits * activation_bits,
respectively.  Other counted tensor contractions use activation_bits**2.

For each mode, four values are reported (in tera-operations = /1e12):
  - Per-forward (full)       : cost of one full DiT forward pass
  - Per-forward (fast step)  : cost of a fast sampling step
                               For uniform schedules: all fast steps skip the
                               same ``cache_num`` blocks.
                               For schedules with a ``cache_schedule`` (produced
                               by gen_dp_slow_steps.py --knapsack or
                               --fixed-blocks): this column shows the *average*
                               fast-step cost across all segments.
  - Avg cost / step          : schedule-averaged cost per sampling step
  - Total cost (N steps)     : avg_per_step * num_sampling_steps

Conventions (matches DiT / ProCache / PTQ4DiT papers):
  - `FLOPs = 2 x MACs`: fvcore's FlopCountAnalysis actually counts MACs
    (one multiply-add = 1 op), and most papers report FLOPs as
    `2 x MACs`. Controlled by --flops-per-mac (default 2.0; set to 1.0
    to get raw MACs).
  - `B = 2` for classifier-free guidance: QuantModel.forward concatenates
    [cond, uncond] into one B=2 forward per sampling step, so DiT / ProCache
    report FLOPs at B=2. Controlled by --batch-size (default 2).

Usage (run from the CacheForDiT/ directory):

  1. Uniform schedule (original):

    python evaluation/measure_flops.py \
        --num-sampling-steps 50 --sampler ddim \
        --cache_start 7 --cache_num 14 --replicate_interval 3

  2. DP / exponential slow-steps (plain step list):

    python evaluation/measure_flops.py \
        --num-sampling-steps 250 --sampler ddim \
        --cache_start 7 --cache_num 14 \
        --slow_steps_path dp_slow_steps.pth

  3. Knapsack / fixed-blocks DP (dict with cache_schedule):

    python evaluation/measure_flops.py \
        --num-sampling-steps 250 --sampler ddim \
        --cache_start 7 --cache_num 14 \
        --slow_steps_path knapsack_solution.pth

    When ``--slow_steps_path`` points to a dict that contains a
    ``cache_schedule``, the script uses the per-segment block counts
    to compute the exact total cost (instead of assuming a uniform
    ``cache_num`` for every fast step).

Does NOT need a checkpoint - MACs/BOPs only depend on architecture, schedule,
and the supplied bit-widths.
Requires: pip install fvcore
"""
import argparse
import logging
import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models import DiT_models  # noqa: E402

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError as e:
    raise ImportError(
        "fvcore is required for MAC/BOP measurement. Install it via:\n"
        "    pip install fvcore\n"
    ) from e


ALL_MODES = ["fp", "quant_only", "cache_only", "quant_cache"]


def _parse_modes(s):
    modes = [m.strip() for m in s.split(",") if m.strip()]
    for m in modes:
        if m not in ALL_MODES:
            raise ValueError(f"Unknown mode: {m}. Valid: {ALL_MODES}")
    return modes


class _FullDiTWrapper(nn.Module):
    """Wrap DiT so fvcore sees a simple (x, t, y) signature with calib=True.

    Passing calib=True skips the per-channel scale-estimation code paths in
    `Attention` / `Mlp` that rely on scipy.stats.spearmanr (CPU code that does
    not trace cleanly with fvcore's JIT-based analyzer).
    """

    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, x, t, y):
        return self.dit(x, t, y, calib=True)


class _BlockWrapper(nn.Module):
    """Wrap a single DiTBlock with (x, c) signature and calib=True."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x, c):
        return self.block(x, c, calib=True)


def count_macs(model, inputs):
    """Return total MACs and MACs grouped by fvcore operator name."""
    # Silence fvcore's per-op warnings; DiT contains LayerNorm / SiLU / GELU
    # which fvcore does not count (consistent with common practice in the
    # transformer FLOPs literature).
    logging.getLogger("fvcore.nn.jit_analysis").setLevel(logging.ERROR)
    analyzer = FlopCountAnalysis(model, inputs)
    analyzer.unsupported_ops_warnings(False)
    analyzer.uncalled_modules_warnings(False)
    return int(analyzer.total()), {
        str(op): int(count) for op, count in analyzer.by_operator().items()
    }


def macs_to_bops(by_operator, fp_bits, weight_bits, act_bits, softmax_bits,
                 quantized):
    """Convert an fvcore MAC profile to bit operations.

    fvcore combines the two equally-sized attention matmuls under ``matmul``.
    Their average bit cost is therefore used here.  ``linear`` and ``conv``
    have a weight operand; remaining contractions are treated as
    activation-activation operations.
    """
    if not quantized:
        return sum(by_operator.values()) * fp_bits * fp_bits

    total = 0.0
    for op, macs in by_operator.items():
        if op in {"linear", "conv"}:
            bit_cost = weight_bits * act_bits
        elif op in {"matmul", "bmm"}:
            bit_cost = 0.5 * (
                act_bits * act_bits + softmax_bits * act_bits
            )
        else:
            bit_cost = act_bits * act_bits
        total += macs * bit_cost
    return int(total)


def schedule_cost(full_cost, block_cost, num_slow, num_fast,
                  cache_num, segment_blocks):
    """Return average fast-step and total schedule cost."""
    fast_uniform = full_cost - cache_num * block_cost
    if segment_blocks is None:
        total = num_slow * full_cost + num_fast * fast_uniform
        return fast_uniform, total

    total = num_slow * full_cost
    for length, blocks in segment_blocks:
        total += length * (full_cost - blocks * block_cost)
    fast_avg = (
        (total - num_slow * full_cost) / num_fast
        if num_fast > 0 else full_cost
    )
    return fast_avg, total


def load_slow_steps_file(path):
    """Load a slow-steps file produced by gen_dp_slow_steps.py (or similar).

    Handles both the legacy format (plain list/tensor of step indices) and the
    new dict format that includes ``cache_schedule`` and ``segments``.

    Returns
    -------
    slow_steps : sorted list[int]
    cache_schedule : dict[int, (cache_start, cache_num)] or None
    """
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        slow_steps = sorted(int(s) for s in raw["slow_steps"])
        cache_schedule = raw.get("cache_schedule", None)
        return slow_steps, cache_schedule
    if torch.is_tensor(raw):
        raw = raw.tolist()
    return sorted(int(s) for s in raw), None


def derive_schedule(num_steps, replicate_interval, slow_steps_path=None,
                    cache_num_default=0):
    """Derive the sampling schedule from either a slow-steps file or a uniform
    replicate_interval.

    Returns
    -------
    num_slow : int
    num_fast : int
    segment_blocks : list[(L, v)] or None
        Per-segment info where *L* = number of fast steps in the segment and
        *v* = number of DiT blocks skipped per fast step.  ``None`` when no
        ``cache_schedule`` is available (uniform ``cache_num_default``).
    """
    cache_schedule = None
    if slow_steps_path and os.path.exists(slow_steps_path):
        slow_steps, cache_schedule = load_slow_steps_file(slow_steps_path)
        slow_steps = sorted(set(slow_steps) | {0})
    else:
        slow_steps = sorted(set(range(0, num_steps, replicate_interval)) | {0})

    num_slow = len(slow_steps)
    num_fast = num_steps - num_slow

    if cache_schedule is not None:
        segment_blocks = []
        for j, s in enumerate(slow_steps):
            end = (slow_steps[j + 1] - 1
                   if j + 1 < len(slow_steps)
                   else num_steps - 1)
            L = end - s
            v = cache_schedule[s][1] if s in cache_schedule else cache_num_default
            segment_blocks.append((L, v))
        return num_slow, num_fast, segment_blocks

    return num_slow, num_fast, None


def main(args):
    modes = _parse_modes(args.modes)

    device = "cuda" if (torch.cuda.is_available() and args.device == "cuda") else "cpu"
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device).eval()

    depth = len(model.blocks)
    hidden = model.blocks[0].attn.qkv.in_features
    num_patches = (latent_size // model.patch_size) ** 2
    B = args.batch_size

    # --- Full-forward MAC profile of the DiT model ---
    x = torch.randn(B, 4, latent_size, latent_size, device=device)
    t = torch.randint(0, 1000, (B,), device=device)
    y = torch.randint(0, args.num_classes, (B,), device=device)

    full_wrapper = _FullDiTWrapper(model).to(device).eval()
    print(f"[info] Counting MACs of one full DiT forward "
          f"(B={B}, image_size={args.image_size}, depth={depth}, hidden={hidden}, "
          f"metric={args.metric})...")
    F_full_macs, full_ops = count_macs(full_wrapper, (x, t, y))

    # --- Single DiTBlock MAC profile (for fast-step savings) ---
    x_tokens = torch.randn(B, num_patches, hidden, device=device)
    c_vec = torch.randn(B, hidden, device=device)
    block_wrapper = _BlockWrapper(model.blocks[0]).to(device).eval()
    print(f"[info] Counting MACs of one DiTBlock "
          f"(tokens={num_patches}, hidden={hidden})...")
    F_block_macs, block_ops = count_macs(block_wrapper, (x_tokens, c_vec))

    # Keep the conventional FLOPs values available for --metric flops.
    F_full = int(F_full_macs * args.flops_per_mac)
    F_block = int(F_block_macs * args.flops_per_mac)

    num_slow, num_fast, segment_blocks = derive_schedule(
        args.num_sampling_steps, args.replicate_interval,
        args.slow_steps_path, args.cache_num)

    if segment_blocks is not None:
        v_active = [v for _, v in segment_blocks if v > 0]
        has_variable_blocks = len(set(v_active)) > 1
    else:
        v_active = []
        has_variable_blocks = False

    # --- Compose per-mode numbers ---
    rows = []
    for mode in modes:
        use_cache = mode in ("cache_only", "quant_cache")
        quantized = mode in ("quant_only", "quant_cache")
        if args.metric == "bops":
            full_cost = macs_to_bops(
                full_ops, args.fp_bits, args.weight_bit, args.act_bit,
                args.sm_abit, quantized)
            block_cost = macs_to_bops(
                block_ops, args.fp_bits, args.weight_bit, args.act_bit,
                args.sm_abit, quantized)
        else:
            full_cost = F_full
            block_cost = F_block

        fast_cached, cache_total = schedule_cost(
            full_cost, block_cost, num_slow, num_fast, args.cache_num,
            segment_blocks)
        if use_cache:
            per_fwd_full = full_cost
            per_fwd_fast = fast_cached
            total = cache_total
            avg_per_step = total / args.num_sampling_steps
        else:
            per_fwd_full = full_cost
            per_fwd_fast = full_cost
            avg_per_step = full_cost
            total = full_cost * args.num_sampling_steps
        rows.append({
            "mode": mode,
            "per_fwd_full_T": per_fwd_full / 1e12,
            "per_fwd_fast_T": per_fwd_fast / 1e12,
            "avg_per_step_T": avg_per_step / 1e12,
            "total_T": total / 1e12,
        })

    # --- Pretty print ---
    print("\n" + "=" * 96)
    metric_label = "BOPs" if args.metric == "bops" else "FLOPs"
    print(f"{metric_label} summary  (B={B}, image_size={args.image_size}, "
          f"steps={args.num_sampling_steps}, sampler={args.sampler}, "
          f"cache_start={args.cache_start}, "
          f"cache_num={args.cache_num}, replicate_interval={args.replicate_interval})")
    if args.metric == "bops":
        print(f"  Convention: BOPs = MACs x operand bit-widths; "
              f"FP={args.fp_bits}-bit, quant=W{args.weight_bit}A{args.act_bit}, "
              f"softmax={args.sm_abit}-bit")
    else:
        print(f"  Convention: B={B} (CFG), FLOPs = {args.flops_per_mac} x MACs "
              f"(fvcore counts MACs)")
    if args.slow_steps_path:
        print(f"  slow_steps_path = {args.slow_steps_path}"
              f"  ({'with cache_schedule' if segment_blocks is not None else 'step list only'})")
    print(f"  slow_steps = {num_slow}, fast_steps = {num_fast}")
    print(f"  Raw full-forward MACs            = {F_full_macs/1e12:.4f} TMACs")
    print(f"  Raw one-block MACs               = {F_block_macs/1e9:.4f} GMACs")
    if segment_blocks is not None and v_active:
        if has_variable_blocks:
            v_min = min(v_active)
            v_max = max(v_active)
            v_avg = sum(v_active) / len(v_active)
            print(f"  blocks skipped / fast-segment   : "
                  f"min={v_min}, max={v_max}, avg={v_avg:.1f}  "
                  f"(variable, from cache_schedule)")
        else:
            print(f"  blocks skipped / fast-segment   : "
                  f"{v_active[0]} (fixed, from cache_schedule)")
    else:
        print(f"  blocks skipped / fast step      : "
              f"{args.cache_num} (uniform)")
    print("=" * 96)
    header = (f"{'Mode':<13}"
              f"{'per-fwd full (T)':>20}"
              f"{'per-fwd fast (T)':>20}"
              f"{'avg/step (T)':>16}"
              f"{'total (T)':>16}")
    print(header)
    print("-" * 96)
    baseline = None
    for r in rows:
        if r["mode"] == "fp":
            baseline = r["total_T"]
            break
    for r in rows:
        line = (f"{r['mode']:<13}"
                f"{r['per_fwd_full_T']:>20.4f}"
                f"{r['per_fwd_fast_T']:>20.4f}"
                f"{r['avg_per_step_T']:>16.4f}"
                f"{r['total_T']:>16.4f}")
        print(line)
    if baseline is not None:
        print("-" * 96)
        print(f"Compression ratio vs fp (total {metric_label}):")
        for r in rows:
            ratio = baseline / r["total_T"] if r["total_T"] > 0 else 0
            print(f"  {r['mode']:<13} {ratio:.3f}x  "
                  f"({metric_label} reduced to {100.0*r['total_T']/baseline:.2f}%)")
    print("=" * 96)
    note = (f"\nNote: BOPs are a hardware-independent compute proxy, not a "
            f"latency prediction; kernel support, memory traffic and quantization "
            f"overheads are not modeled. Sampler={args.sampler}.")
    if args.metric == "flops":
        note = (f"\nNote: FLOPs are bit-width independent, so fp == quant_only "
                f"and cache_only == quant_cache. Sampler={args.sampler}.")
    if segment_blocks is not None and has_variable_blocks:
        note += (f"\n      per-fwd fast (T) shows the *average* across "
                 f"segments with variable block counts.")
    print(note)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    p.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--num-sampling-steps", type=int, default=50)
    p.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"],
                   help="Sampling algorithm: ddpm (stochastic) or ddim (deterministic when eta=0). "
                        "FLOPs per forward are identical; this flag is recorded for bookkeeping.")
    p.add_argument("--eta", type=float, default=0.0,
                   help="DDIM eta parameter (0=deterministic, 1=fully stochastic like DDPM)")
    p.add_argument("--batch-size", type=int, default=2,
                   help="Batch size used for FLOPs counting. DiT / ProCache / PTQ4DiT "
                        "typically report at B=2 to account for classifier-free guidance "
                        "(QuantModel.forward concatenates [cond, uncond] into one B=2 "
                        "forward per sampling step). Use B=1 to report single-image FLOPs "
                        "without CFG.")
    p.add_argument("--flops-per-mac", type=float, default=2.0,
                   help="FLOPs per MAC. fvcore counts MACs (one multiply-add = 1 op); "
                        "most DiT / ProCache / PTQ4DiT papers report FLOPs = 2 x MACs. "
                        "Set to 1.0 to report raw MACs instead.")
    p.add_argument("--metric", choices=["bops", "flops"], default="bops",
                   help="Report bit operations (default) or conventional FLOPs.")
    p.add_argument("--fp-bits", type=int, default=16,
                   help="Operand width for the floating-point baseline.")
    p.add_argument("--weight-bit", type=int, default=8,
                   help="Quantized weight width (matches quant_cache_sample.py).")
    p.add_argument("--act-bit", type=int, default=8,
                   help="Quantized activation width.")
    p.add_argument("--sm-abit", type=int, default=8,
                   help="Quantized post-softmax attention width.")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    p.add_argument("--modes", type=str, default=",".join(ALL_MODES),
                   help="Comma-separated list of modes to tabulate.")

    p.add_argument("--replicate_interval", type=int, default=3)
    p.add_argument("--cache_start", type=int, default=7)
    p.add_argument("--cache_num", type=int, default=14)
    p.add_argument("--slow_steps_path", type=str, default=None)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
