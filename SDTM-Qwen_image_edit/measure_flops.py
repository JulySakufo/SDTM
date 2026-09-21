"""Measure Qwen transformer FLOPs with identical edit inputs for all modes."""

from __future__ import annotations

import copy
import gc
import types
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from PIL import Image
from torch.utils.flop_counter import FlopCounterMode

from evaluation_utils import write_json
from qwen_core import (
    install_counting_processors,
    patch_statistics,
    restore_counting_processors,
    set_batch_seeds,
)
from sample import (
    MODES,
    apply_mode,
    build_parser as build_sample_parser,
    dimension_bucket,
    image_filename,
    image_seed,
    load_captions,
    load_pipeline,
    pipeline_call_kwargs,
    validate_args,
)


@dataclass
class CallRecord:
    call_index: int
    denoising_step: int
    cfg_branch: str
    kind: str
    flops: int
    by_operator: Dict[str, int]
    image_tokens: int
    img_shapes: Any


def parse_modes(value: str) -> List[str]:
    aliases = {mode.lower(): mode for mode in MODES}
    aliases.update(
        baseline="Default",
        tome="ToMe",
        testidm2="SDTM_testidm2",
        taylorseer="SDTM_TaylorSeer",
    )
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
    def __init__(self, transformer, mode: str, calls_per_step: int):
        self.transformer = transformer
        self.mode = mode
        self.calls_per_step = calls_per_step
        self.original = transformer.forward
        self.records: List[CallRecord] = []
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
            operators = {
                str(operator): int(value) for operator, value in counts.items()
            }
            index = len(recorder.records)
            state = getattr(module, "_qwen_sdtm_info", {}).get("state", {})
            recorder.records.append(
                CallRecord(
                    call_index=index,
                    denoising_step=int(
                        state.get("step", index // recorder.calls_per_step)
                    ),
                    cfg_branch=str(
                        state.get(
                            "cfg_branch",
                            "uncond"
                            if recorder.calls_per_step == 2 and index % 2
                            else "cond",
                        )
                    ),
                    kind=str(state.get("taylor_type", "full")),
                    flops=sum(operators.values()),
                    by_operator=operators,
                    image_tokens=hidden.shape[1],
                    img_shapes=kwargs.get("img_shapes"),
                )
            )
            return output

        self.transformer.forward = types.MethodType(counted, self.transformer)

    def uninstall(self) -> None:
        self.transformer.forward = self.original


def select_input(args):
    captions = load_captions(args.caption_path)
    if args.flops_image_id is not None:
        candidates = [
            item for item in captions if int(item["image_id"]) == args.flops_image_id
        ]
        if not candidates:
            raise ValueError(
                f"image_id {args.flops_image_id} is not present in the caption file"
            )
        item = candidates[0]
    else:
        item = captions[0]
    path = Path(args.input_image_dir).expanduser().resolve() / image_filename(
        item["image_id"]
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return item, path


def measure(mode: str, base_args, item, image_path: Path) -> Dict[str, Any]:
    args = copy.copy(base_args)
    args.tore_type = mode
    validate_args(args)
    pipe = apply_mode(load_pipeline(args), args)
    saved_processors = install_counting_processors(pipe.transformer)
    calls_per_step = (
        2 if args.true_cfg_scale > 1 and args.negative_prompt is not None else 1
    )
    recorder = TransformerRecorder(pipe.transformer, mode, calls_per_step)
    recorder.install()
    try:
        target_height, target_width, reference_height, reference_width = (
            dimension_bucket(image_path, args.height, args.width)
        )
        with Image.open(image_path) as opened:
            source = opened.convert("RGB")
        seeds = [image_seed(args.seed, item["image_id"])] * args.batch_size
        generators = [torch.Generator("cpu").manual_seed(seed) for seed in seeds]
        prompts = [
            args.prompt if args.prompt is not None else item["caption"]
        ] * args.batch_size
        images = [source.copy() for _ in range(args.batch_size)]
        set_batch_seeds(pipe, seeds)
        with torch.no_grad():
            output = pipe(
                **pipeline_call_kwargs(
                    args,
                    prompts,
                    images,
                    generators,
                    target_height,
                    target_width,
                ),
                output_type="latent",
            )
        del output
        expected_calls = args.num_inference_steps * calls_per_step
        if len(recorder.records) != expected_calls:
            raise RuntimeError(
                f"expected {expected_calls} transformer calls, recorded {len(recorder.records)}"
            )
        totals, by_branch, by_kind = (
            defaultdict(int),
            defaultdict(int),
            defaultdict(int),
        )
        for record in recorder.records:
            by_branch[record.cfg_branch] += record.flops
            by_kind[record.kind] += record.flops
            for operator, value in record.by_operator.items():
                totals[operator] += value
        total = sum(record.flops for record in recorder.records)
        return {
            "mode": mode,
            "input_image_id": item["image_id"],
            "prompt": prompts[0],
            "target_size": [target_height, target_width],
            "reference_size": [reference_height, reference_width],
            "num_transformer_calls": len(recorder.records),
            "num_denoising_steps": args.num_inference_steps,
            "hidden_states_shape": recorder.hidden_shape,
            "encoder_hidden_states_shape": recorder.encoder_shape,
            "flops_per_transformer_call": total / len(recorder.records),
            "flops_per_denoising_step": total / args.num_inference_steps,
            "flops_per_edited_image": total / args.batch_size,
            "total_batch_flops": total,
            "by_cfg_branch": dict(by_branch),
            "by_execution_kind": dict(by_kind),
            "by_operator": dict(
                sorted(totals.items(), key=lambda pair: pair[1], reverse=True)
            ),
            "patch": patch_statistics(pipe),
            "calls": [asdict(record) for record in recorder.records],
        }
    finally:
        recorder.uninstall()
        restore_counting_processors(saved_processors)
        recorder.original = None
        recorder.transformer = None
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def add_relative_reductions(results: List[Dict[str, Any]]) -> None:
    baseline = next((result for result in results if result["mode"] == "Default"), None)
    if baseline is None:
        return
    base = baseline["flops_per_edited_image"]
    for result in results:
        result["relative_flops_reduction_vs_default"] = (
            1.0 - result["flops_per_edited_image"] / base
        )


def print_summary(results: List[Dict[str, Any]]) -> None:
    print("\nQwen transformer FLOPs (multiply-add = 2 FLOPs)")
    print(
        f"{'Mode':<24}{'TFLOPs/call':>16}{'TFLOPs/step':>16}{'TFLOPs/image':>18}{'reduction':>13}"
    )
    for result in results:
        reduction = result.get("relative_flops_reduction_vs_default")
        reduction_text = "-" if reduction is None else f"{100 * reduction:.2f}%"
        print(
            f"{result['mode']:<24}{result['flops_per_transformer_call'] / 1e12:>16.4f}"
            f"{result['flops_per_denoising_step'] / 1e12:>16.4f}"
            f"{result['flops_per_edited_image'] / 1e12:>18.4f}{reduction_text:>13}"
        )


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_sample_parser()
    parser.description = __doc__
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument(
        "--prompt", default=None, help="Override the selected COCO caption"
    )
    parser.add_argument("--flops-image-id", type=int, default=None)
    parser.add_argument("--output-json", default="flops_qwen_image_edit.json")
    args = parser.parse_args(argv)
    args.warmup_runs = 0
    args.show_pipeline_progress = False
    modes = parse_modes(args.modes)
    item, image_path = select_input(args)
    results = [measure(mode, args, item, image_path) for mode in modes]
    add_relative_reductions(results)
    print_summary(results)
    write_json(
        args.output_json,
        {
            "configuration": {
                "model_path": args.model_path,
                "input_image": str(image_path),
                "steps": args.num_inference_steps,
                "true_cfg_scale": args.true_cfg_scale,
                "batch_size": args.batch_size,
                "dtype": args.torch_dtype,
                "counted_region": "QwenImageTransformer2DModel only",
                "modes": modes,
                "parameters": vars(args),
            },
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
