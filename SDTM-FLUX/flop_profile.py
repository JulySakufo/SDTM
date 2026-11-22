"""Count the FLOPs executed by the FLUX transformer in one denoising run."""

from __future__ import annotations

import types
from collections import defaultdict
from dataclasses import asdict, dataclass

import torch
from torch.utils.flop_counter import FlopCounterMode

from flux_core import patch_statistics


COUNTED_REGION = "FluxTransformer2DModel only"


@dataclass
class StepRecord:
    call_index: int
    kind: str
    flops: int
    by_operator: dict[str, int]


def attention_flops(query, key, value, *args, **kwargs) -> int:
    batch, heads, queries, query_dim = query
    keys, value_dim = key[-2], value[-1]
    return 2 * batch * heads * queries * keys * (query_dim + value_dim)


def profile_transformer(pipe, mode: str, call_kwargs: dict) -> dict:
    """Replay one batch with FLOP counting and omit VAE decoding from the replay."""
    transformer = pipe.transformer
    original_forward = transformer.forward
    instance_forward = vars(transformer).get("forward")
    records: list[StepRecord] = []
    hidden_shape = encoder_shape = None
    attention_ops = {
        operator: attention_flops
        for name in ("_scaled_dot_product_flash_attention_for_cpu", "_scaled_dot_product_cudnn_attention")
        if (operator := getattr(torch.ops.aten, name, None)) is not None
    }

    def counted(module, *args, **kwargs):
        nonlocal hidden_shape, encoder_shape
        if not records:
            hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            hidden_shape = list(hidden.shape)
            encoder_shape = list(kwargs["encoder_hidden_states"].shape)
        with FlopCounterMode(display=False, custom_mapping=attention_ops) as counter:
            output = original_forward(*args, **kwargs)
        operators = {str(operator): int(value) for operator, value in counter.get_flop_counts().get("Global", {}).items()}
        state = getattr(module, "_flux_sdtm_info", {}).get("state", {})
        kind = state.get("taylor_type", "full") if mode in {"Default", "SDTM_TaylorSeer"} else mode
        records.append(StepRecord(len(records), kind, sum(operators.values()), operators))
        return output

    transformer.forward = types.MethodType(counted, transformer)
    try:
        # PyTorch 2.1 bypasses the dispatch counter under inference_mode.
        with torch.no_grad():
            pipe(**call_kwargs, output_type="latent")
    finally:
        if instance_forward is None:
            del transformer.forward
        else:
            transformer.forward = instance_forward

    expected_steps = call_kwargs["num_inference_steps"]
    if len(records) != expected_steps:
        raise RuntimeError(f"expected {expected_steps} transformer calls, recorded {len(records)}")
    totals: dict[str, int] = defaultdict(int)
    for record in records:
        for operator, count in record.by_operator.items():
            totals[operator] += count
    total = sum(totals.values())
    if not total:
        raise RuntimeError("the FLOP counter recorded no transformer operations")
    image_count = len(call_kwargs["prompt"])
    return {
        "mode": mode,
        "counted_region": COUNTED_REGION,
        "profiled_images": image_count,
        "num_transformer_calls": len(records),
        "hidden_states_shape": hidden_shape,
        "encoder_hidden_states_shape": encoder_shape,
        "total_flops": total,
        "average_flops_per_step": total / len(records),
        "total_flops_per_image": total / image_count,
        "patch": patch_statistics(pipe),
        "by_operator": dict(sorted(totals.items(), key=lambda item: item[1], reverse=True)),
        "steps": [asdict(record) for record in records],
    }
