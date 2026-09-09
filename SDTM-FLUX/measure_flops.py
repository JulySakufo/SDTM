"""Measure executed FLUX transformer FLOPs for all SDTM migration modes."""

from __future__ import annotations

import gc
import json
import types
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List

import torch
from torch.utils.flop_counter import FlopCounterMode

from evaluation_utils import write_json
from flux_core import patch_statistics, set_batch_seeds
from sample import MODES, apply_mode, build_parser as build_sample_parser, image_seed, load_pipeline, validate_args


@dataclass
class StepRecord:
    call_index: int
    kind: str
    flops: int
    by_operator: Dict[str, int]


def parse_modes(value: str) -> List[str]:
    aliases = {mode.lower(): mode for mode in MODES}
    aliases.update(baseline="Default", tome="ToMe", testidm2="SDTM_testidm2", taylorseer="SDTM_TaylorSeer")
    result = []
    for raw in value.split(","):
        if not raw.strip():
            continue
        canonical = aliases.get(raw.strip().lower())
        if canonical is None:
            raise ValueError(f"unknown mode {raw!r}; choose from {', '.join(MODES)}")
        if canonical not in result:
            result.append(canonical)
    if not result:
        raise ValueError("at least one FLOP mode is required")
    return result


class TransformerRecorder:
    def __init__(self, transformer, mode: str):
        self.transformer = transformer
        self.mode = mode
        self.original = transformer.forward
        self.records: List[StepRecord] = []
        self.hidden_shape = None
        self.encoder_shape = None

    def install(self) -> None:
        recorder, original = self, self.original

        def counted(module, *args, **kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            encoder = kwargs.get("encoder_hidden_states")
            if recorder.hidden_shape is None:
                recorder.hidden_shape = list(hidden.shape)
                recorder.encoder_shape = list(encoder.shape)
            with FlopCounterMode(display=False) as counter:
                output = original(*args, **kwargs)
            counts = counter.get_flop_counts().get("Global", {})
            operators = {str(operator): int(value) for operator, value in counts.items()}
            info = getattr(module, "_flux_sdtm_info", {})
            state = info.get("state", {})
            kind = state.get("taylor_type", "full")
            if recorder.mode not in {"Default", "SDTM_TaylorSeer"}:
                kind = recorder.mode
            recorder.records.append(StepRecord(len(recorder.records), kind, sum(operators.values()), operators))
            return output

        self.transformer.forward = types.MethodType(counted, self.transformer)

    def uninstall(self) -> None:
        self.transformer.forward = self.original


def measure(mode: str, args) -> Dict[str, Any]:
    args.tore_type = mode
    validate_args(args)
    pipe = apply_mode(load_pipeline(args), args)
    recorder = TransformerRecorder(pipe.transformer, mode)
    recorder.install()
    try:
        seeds = [image_seed(args.seed, index) for index in range(args.batch_size)]
        set_batch_seeds(pipe, seeds)
        generators = [torch.Generator("cpu").manual_seed(seed) for seed in seeds]
        prompts = [args.prompt] * args.batch_size
        with torch.no_grad():
            output = pipe(
                prompt=prompts,
                generator=generators,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                max_sequence_length=args.max_sequence_length,
                output_type="latent",
            )
        del output
        if not recorder.records:
            raise RuntimeError("the FLUX transformer was not called")
        if len(recorder.records) != args.num_inference_steps:
            raise RuntimeError(
                f"expected {args.num_inference_steps} transformer calls, recorded {len(recorder.records)}"
            )
        totals = defaultdict(int)
        for record in recorder.records:
            for operator, value in record.by_operator.items():
                totals[operator] += value
        total = sum(record.flops for record in recorder.records)
        return {
            "mode": mode,
            "num_transformer_calls": len(recorder.records),
            "hidden_states_shape": recorder.hidden_shape,
            "encoder_hidden_states_shape": recorder.encoder_shape,
            "total_flops": total,
            "average_flops_per_step": total / len(recorder.records),
            "total_flops_per_image": total / args.batch_size,
            "patch": patch_statistics(pipe),
            "by_operator": dict(sorted(totals.items(), key=lambda item: item[1], reverse=True)),
            "steps": [asdict(record) for record in recorder.records],
        }
    finally:
        recorder.uninstall()
        recorder.original = None
        recorder.transformer = None
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def print_summary(results: List[Dict[str, Any]]) -> None:
    baseline = next((result for result in results if result["mode"] == "Default"), None)
    print("\nFLUX transformer FLOPs (multiply-add = 2 FLOPs)")
    print(f"{'Mode':<24}{'TFLOPs/step':>16}{'TFLOPs/image':>18}{'vs Default':>14}")
    for result in results:
        speedup = "-"
        if baseline and result["total_flops_per_image"]:
            speedup = f"{baseline['total_flops_per_image'] / result['total_flops_per_image']:.3f}x"
        print(
            f"{result['mode']:<24}{result['average_flops_per_step']/1e12:>16.4f}"
            f"{result['total_flops_per_image']/1e12:>18.4f}{speedup:>14}"
        )


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_sample_parser()
    parser.description = __doc__
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--prompt", default="A photo of a cat")
    parser.add_argument("--output-json", default="flops_flux.json")
    args = parser.parse_args(argv)
    args.warmup_runs = 0
    args.show_pipeline_progress = False
    modes = parse_modes(args.modes)
    results = [measure(mode, args) for mode in modes]
    print_summary(results)
    write_json(
        args.output_json,
        {
            "configuration": {
                "model_path": args.model_path,
                "height": args.height,
                "width": args.width,
                "steps": args.num_inference_steps,
                "batch_size": args.batch_size,
                "dtype": args.torch_dtype,
                "counted_region": "FluxTransformer2DModel only",
                "modes": modes,
                "parameters": vars(args),
            },
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
