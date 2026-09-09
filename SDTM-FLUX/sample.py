"""Unified COCO sampler for FLUX.1-dev and the four acceleration modes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from tqdm import tqdm

from flux_core import patch_statistics, set_batch_seeds


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = (HERE / "../../models/FLUX.1-dev").resolve()
DEFAULT_CAPTIONS = (HERE / "../datasets/COCO2017/captions_val2017.json").resolve()
MODES = ("Default", "ToMe", "SDTM", "SDTM_testidm2", "SDTM_TaylorSeer")


def load_captions(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    """Read COCO annotations without modifying the source JSON."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    annotations = data.get("annotations", []) if isinstance(data, dict) else data
    if not isinstance(annotations, list):
        raise ValueError("caption JSON must be a list or contain an 'annotations' list")
    longest: Dict[Any, Dict[str, Any]] = {}
    for item in annotations:
        image_id, caption = item["image_id"], str(item["caption"])
        if image_id not in longest or len(caption) > len(longest[image_id]["caption"]):
            longest[image_id] = {"image_id": image_id, "caption": caption}
    return list(longest.values())


def image_seed(base_seed: int, image_id: Any) -> int:
    try:
        identifier = int(image_id)
    except (TypeError, ValueError):
        digest = hashlib.sha256(str(image_id).encode("utf-8")).digest()
        identifier = int.from_bytes(digest[:8], "little")
    return (int(base_seed) + identifier) % (2**63 - 1)


def image_filename(image_id: Any) -> str:
    value = str(image_id)
    return f"{int(value):012d}.jpg" if value.isdigit() else f"{value}.jpg"


def _dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def load_pipeline(args):
    from diffusers import FluxPipeline

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible. FLUX.1-dev is too large for the default CPU sampling path.")
    pipe = FluxPipeline.from_pretrained(
        args.model_path,
        torch_dtype=_dtype(args.torch_dtype),
        local_files_only=args.local_files_only,
    )
    if args.cpu_offload:
        if not args.device.startswith("cuda"):
            raise ValueError("--cpu-offload requires a CUDA device")
        gpu_id = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
        pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    else:
        pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=not args.show_pipeline_progress)
    pipe.transformer.eval()
    return pipe


def apply_mode(pipe, args):
    if args.tore_type == "Default":
        return pipe
    if args.tore_type == "ToMe":
        from TR_ToMe import apply_ToMe

        return apply_ToMe(
            pipe,
            ratio=args.ToMe_ratio,
            sx=args.ToMe_sx,
            sy=args.ToMe_sy,
            use_rand=args.ToMe_use_rand,
            merge_attn=args.ToMe_merge_attn,
            merge_mlp=args.ToMe_merge_mlp,
            change_merge_to_prune=args.ToMe_pseudo_merge,
            seed=args.seed,
        )

    common = dict(
        ratio=args.SDTM_ratio,
        deviation=args.SDTM_deviation,
        switch_step=args.SDTM_switch_step,
        use_rand=args.SDTM_use_rand,
        sx=args.SDTM_sx,
        sy=args.SDTM_sy,
        a_s=args.SDTM_a_s,
        a_d=args.SDTM_a_d,
        a_p=args.SDTM_a_p,
        pseudo_merge=args.SDTM_pseudo_merge,
        mcw=args.SDTM_mcw,
        protect_steps_frequency=args.SDTM_protect_steps_frequency,
        protect_layers_frequency=args.SDTM_protect_layers_frequency,
        cache_each_step=args.SDTM_cache_each_step,
        seed=args.seed,
    )
    if args.tore_type == "SDTM":
        from TR_SDTM import apply_SDTM

        return apply_SDTM(pipe, **common)

    adaptive = dict(
        adaptive_ssm=args.SDTM_adaptive_ssm,
        ssm_threshold=args.SDTM_ssm_threshold,
        low_complexity_threshold=args.SDTM_low_complexity_threshold,
        high_complexity_threshold=args.SDTM_high_complexity_threshold,
    )
    if args.tore_type == "SDTM_testidm2":
        from TR_SDTM_testidm2 import apply_SDTM

        return apply_SDTM(pipe, **common, **adaptive, slic_alpha=args.SDTM_slic_alpha)
    if args.tore_type == "SDTM_TaylorSeer":
        from TR_SDTM_TaylorSeer import apply_SDTM_TaylorSeer

        return apply_SDTM_TaylorSeer(
            pipe,
            **common,
            **adaptive,
            taylor_interval=args.Taylor_interval,
            taylor_max_order=args.Taylor_max_order,
            taylor_first_enhance=args.Taylor_first_enhance,
        )
    raise ValueError(f"unsupported mode: {args.tore_type}")


def experiment_name(args) -> str:
    if args.experiment_name:
        return args.experiment_name
    base = (
        f"FLUX1-dev-{args.tore_type}-{args.height}x{args.width}-"
        f"steps{args.num_inference_steps}-g{args.guidance_scale:g}-seed{args.seed}"
    )
    if args.tore_type == "ToMe":
        return (
            f"{base}-R{args.ToMe_ratio:g}-{args.ToMe_sy}x{args.ToMe_sx}-"
            f"rnd{int(args.ToMe_use_rand)}-A{int(args.ToMe_merge_attn)}-"
            f"M{int(args.ToMe_merge_mlp)}-PM{int(args.ToMe_pseudo_merge)}"
        )
    if args.tore_type.startswith("SDTM"):
        base += (
            f"-R{args.SDTM_ratio:g}-D{args.SDTM_deviation:g}-Sw{args.SDTM_switch_step}-"
            f"{args.SDTM_sy}x{args.SDTM_sx}-rnd{int(args.SDTM_use_rand)}-"
            f"as{args.SDTM_a_s:g}-ad{args.SDTM_a_d:g}-ap{args.SDTM_a_p:g}-"
            f"PM{int(args.SDTM_pseudo_merge)}-W{args.SDTM_mcw:g}-"
            f"Ps{args.SDTM_protect_steps_frequency}-Pl{args.SDTM_protect_layers_frequency}-"
            f"CE{int(args.SDTM_cache_each_step)}"
        )
    if args.tore_type in {"SDTM_testidm2", "SDTM_TaylorSeer"}:
        base += (
            f"-Adp{int(args.SDTM_adaptive_ssm)}-Th{args.SDTM_ssm_threshold:g}-"
            f"L{args.SDTM_low_complexity_threshold:g}-H{args.SDTM_high_complexity_threshold:g}"
        )
    if args.tore_type == "SDTM_testidm2":
        base += f"-SLIC{args.SDTM_slic_alpha:g}"
    if args.tore_type == "SDTM_TaylorSeer":
        base += f"-TI{args.Taylor_interval}-TO{args.Taylor_max_order}-TE{args.Taylor_first_enhance}"
    return base


def pipeline_call_kwargs(args, prompts: Sequence[str], generators: Sequence[torch.Generator]) -> Dict[str, Any]:
    return {
        "prompt": list(prompts),
        "generator": list(generators),
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "max_sequence_length": args.max_sequence_length,
    }


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
    temporary.replace(path)


def run(args) -> Path:
    captions = load_captions(args.caption_path)
    if args.limit is not None:
        captions = captions[: args.limit]
    output_dir = Path(args.output_path).expanduser().resolve() / experiment_name(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = apply_mode(load_pipeline(args), args)
    print(f"Mode: {args.tore_type}; patch: {patch_statistics(pipe)}")
    print(f"Output: {output_dir}; captions: {len(captions)}")

    if args.warmup_runs:
        warmup_seeds = [args.seed]
        set_batch_seeds(pipe, warmup_seeds)
        warmup_generators = [torch.Generator("cpu").manual_seed(args.seed)]
        for _ in range(args.warmup_runs):
            with torch.inference_mode():
                pipe(**pipeline_call_kwargs(args, ["A photo of a cat"], warmup_generators))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    generated = skipped = 0
    started = time.perf_counter()
    for start in tqdm(range(0, len(captions), args.batch_size), desc="Generating images"):
        batch = captions[start : start + args.batch_size]
        if args.skip_existing:
            remaining = [item for item in batch if not (output_dir / image_filename(item["image_id"])).exists()]
            skipped += len(batch) - len(remaining)
            batch = remaining
        if not batch:
            continue
        seeds = [image_seed(args.seed, item["image_id"]) for item in batch]
        generators = [torch.Generator("cpu").manual_seed(seed) for seed in seeds]
        set_batch_seeds(pipe, seeds)
        with torch.inference_mode():
            images = pipe(
                **pipeline_call_kwargs(args, [item["caption"] for item in batch], generators)
            ).images
        for item, image in zip(batch, images):
            image.save(output_dir / image_filename(item["image_id"]), quality=95)
            generated += 1

    elapsed = time.perf_counter() - started
    expected = [image_filename(item["image_id"]) for item in captions]
    missing = [name for name in expected if not (output_dir / name).is_file()]
    manifest = {
        "mode": args.tore_type,
        "model_path": str(Path(args.model_path).resolve()),
        "parameters": vars(args),
        "patch": patch_statistics(pipe),
        "caption_count": len(captions),
        "generated_this_run": generated,
        "skipped_existing": skipped,
        "missing_count": len(missing),
        "missing_files": missing[:100],
        "elapsed_seconds": elapsed,
        "seconds_per_expected_image": elapsed / max(len(captions), 1),
        "torch_version": torch.__version__,
    }
    try:
        import diffusers

        manifest["diffusers_version"] = diffusers.__version__
    except ImportError:
        pass
    _write_json(output_dir / "manifest.json", manifest)
    if missing:
        raise RuntimeError(f"generation finished with {len(missing)} missing images; see manifest.json")
    print(f"Completed {len(captions)} images in {elapsed:.2f}s ({generated} newly generated).")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caption-path", default=str(DEFAULT_CAPTIONS))
    parser.add_argument("--output-path", default="../../../../irip_16t_0/huangyu_2026/samples")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--torch-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tore-type", choices=MODES, default="SDTM")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--show-pipeline-progress", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--ToMe-ratio", type=float, default=0.9)
    parser.add_argument("--ToMe-sx", type=int, default=2)
    parser.add_argument("--ToMe-sy", type=int, default=2)
    parser.add_argument("--ToMe-use-rand", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ToMe-merge-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ToMe-merge-mlp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ToMe-pseudo-merge", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--SDTM-ratio", type=float, default=0.3)
    parser.add_argument("--SDTM-deviation", type=float, default=0.2)
    parser.add_argument("--SDTM-switch-step", type=int, default=20)
    parser.add_argument("--SDTM-use-rand", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--SDTM-sx", type=int, default=4)
    parser.add_argument("--SDTM-sy", type=int, default=4)
    parser.add_argument("--SDTM-a-s", type=float, default=0.05)
    parser.add_argument("--SDTM-a-d", type=float, default=0.05)
    parser.add_argument("--SDTM-a-p", type=float, default=2.0)
    parser.add_argument("--SDTM-adaptive-ssm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--SDTM-ssm-threshold", type=float, default=0.0)
    parser.add_argument("--SDTM-low-complexity-threshold", type=float, default=0.66)
    parser.add_argument("--SDTM-high-complexity-threshold", type=float, default=0.33)
    parser.add_argument("--SDTM-pseudo-merge", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--SDTM-mcw", type=float, default=0.1)
    parser.add_argument("--SDTM-protect-steps-frequency", type=int, default=3)
    parser.add_argument("--SDTM-protect-layers-frequency", type=int, default=-1)
    parser.add_argument("--SDTM-cache-each-step", dest="SDTM_cache_each_step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--SDTM-slic-alpha", type=float, default=0.5)

    parser.add_argument("--Taylor-interval", type=int, default=2)
    parser.add_argument("--Taylor-max-order", type=int, default=1)
    parser.add_argument("--Taylor-first-enhance", type=int, default=12)
    return parser


def validate_args(args) -> None:
    if args.batch_size <= 0 or args.num_inference_steps <= 0:
        raise ValueError("batch size and inference steps must be positive")
    if args.height <= 0 or args.width <= 0 or args.height % 16 or args.width % 16:
        raise ValueError("FLUX height and width must be positive multiples of 16")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.tore_type == "ToMe" and not 0 <= args.ToMe_ratio <= 1:
        raise ValueError("--ToMe-ratio must be in [0, 1]")
    grid_height, grid_width = args.height // 16, args.width // 16
    if args.tore_type == "ToMe":
        if args.ToMe_sx <= 0 or args.ToMe_sy <= 0 or grid_height % args.ToMe_sy or grid_width % args.ToMe_sx:
            raise ValueError(
                f"FLUX token grid {grid_height}x{grid_width} must be divisible by "
                f"ToMe window {args.ToMe_sy}x{args.ToMe_sx}"
            )
    if args.tore_type.startswith("SDTM"):
        if not 0 <= args.SDTM_ratio <= 1:
            raise ValueError("--SDTM-ratio must be in [0, 1]")
        if args.SDTM_deviation < 0 or not 0 <= args.SDTM_ratio - args.SDTM_deviation <= 1 or not 0 <= args.SDTM_ratio + args.SDTM_deviation <= 1:
            raise ValueError("the complete SDTM ratio ± deviation schedule must remain in [0, 1]")
        if args.SDTM_switch_step < 0 or args.SDTM_switch_step >= args.num_inference_steps:
            raise ValueError("--SDTM-switch-step must identify an inference step")
        if args.SDTM_sx <= 0 or args.SDTM_sy <= 0 or grid_height % args.SDTM_sy or grid_width % args.SDTM_sx:
            raise ValueError(
                f"FLUX token grid {grid_height}x{grid_width} must be divisible by "
                f"SDTM window {args.SDTM_sy}x{args.SDTM_sx}"
            )
        if args.tore_type in {"SDTM_testidm2", "SDTM_TaylorSeer"} and args.SDTM_sx * args.SDTM_sy < 4:
            raise ValueError("multi-level SSM requires at least four tokens per window")
    if args.Taylor_interval <= 0 or args.Taylor_max_order < 0 or args.Taylor_first_enhance < 0:
        raise ValueError("invalid TaylorSeer schedule")


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    validate_args(args)
    run(args)


if __name__ == "__main__":
    main()
