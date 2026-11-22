"""Measure executed FLUX transformer FLOPs for all SDTM migration modes."""

from __future__ import annotations

import gc
from copy import copy
from typing import Any, Dict, Iterable, List

import torch

from evaluation_utils import write_json
from flop_profile import COUNTED_REGION, profile_transformer
from sample import apply_mode, batch_call_kwargs, build_parser, load_captions, load_pipeline, parse_modes, validate_args


def measure(mode: str, args) -> Dict[str, Any]:
    args = copy(args)
    args.tore_type = mode
    validate_args(args)
    if args.caption_path is not None:
        captions = load_captions(args.caption_path)[: args.limit]
    else:
        prompts = args.prompt or ["A photo of a cat"]
        if len(prompts) == 1:
            prompts = prompts * args.batch_size
        captions = [{"image_id": index, "caption": prompt} for index, prompt in enumerate(prompts)]
    batch = captions[: args.batch_size]
    if not batch:
        raise ValueError("the selected input contains no prompts")
    pipe = apply_mode(load_pipeline(args), args)
    try:
        return profile_transformer(pipe, mode, batch_call_kwargs(pipe, args, batch))
    finally:
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
    parser = build_parser(default_mode=None)
    parser.description = __doc__
    parser.add_argument("--output-json", default="flops_flux.json")
    args = parser.parse_args(argv)
    args.warmup_runs = 0
    args.show_pipeline_progress = False
    modes = [args.tore_type] if args.tore_type is not None else parse_modes(args.modes or "all")
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
                "counted_region": COUNTED_REGION,
                "modes": modes,
                "parameters": vars(args),
            },
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
