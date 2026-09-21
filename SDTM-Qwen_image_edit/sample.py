"""Unified COCO sampler for Qwen-Image-Edit and four acceleration modes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from qwen_core import (
    install_qwen_sdpa_compatibility,
    patch_statistics,
    set_batch_seeds,
)


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = Path("/irip_16t_0/huangyu_2026/models/Qwen-Image-Edit")
DEFAULT_CAPTIONS = (HERE / "../datasets/COCO2017/captions_val2017.json").resolve()
DEFAULT_INPUT_IMAGES = (HERE / "../datasets/val2017").resolve()
DEFAULT_OUTPUT = Path("/irip_16t_0/huangyu_2026/samples")
MODES = ("Default", "ToMe", "SDTM", "SDTM_testidm2", "SDTM_TaylorSeer")


def load_captions(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    """Return one (longest) caption per image without modifying annotations."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    annotations = data.get("annotations", []) if isinstance(data, dict) else data
    if not isinstance(annotations, list):
        raise ValueError("caption JSON must be a list or contain an 'annotations' list")
    longest: "OrderedDict[Any, Dict[str, Any]]" = OrderedDict()
    for item in annotations:
        image_id, caption = item["image_id"], str(item["caption"])
        if image_id not in longest or len(caption) > len(longest[image_id]["caption"]):
            longest[image_id] = {"image_id": image_id, "caption": caption}
    return list(longest.values())


def image_seed(base_seed: int, image_id: Any) -> int:
    try:
        identifier = int(image_id)
    except (TypeError, ValueError):
        identifier = int.from_bytes(
            hashlib.sha256(str(image_id).encode("utf-8")).digest()[:8], "little"
        )
    return (int(base_seed) + identifier) % (2**63 - 1)


def image_filename(image_id: Any) -> str:
    value = str(image_id)
    return f"{int(value):012d}.jpg" if value.isdigit() else f"{value}.jpg"


def calculate_dimensions(target_area: int, ratio: float) -> Tuple[int, int]:
    """Match Diffusers 0.35.1 Qwen-Image-Edit's 1MP size quantization."""
    width = round(math.sqrt(target_area * ratio) / 32) * 32
    height = round((math.sqrt(target_area * ratio) / ratio) / 32) * 32
    return int(width), int(height)


def image_dimensions(path: Path) -> Tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def dimension_bucket(
    path: Path, height: int | None, width: int | None
) -> Tuple[int, int, int, int]:
    source_width, source_height = image_dimensions(path)
    reference_width, reference_height = calculate_dimensions(
        1024 * 1024, source_width / source_height
    )
    target_height = reference_height if height is None else height
    target_width = reference_width if width is None else width
    return target_height, target_width, reference_height, reference_width


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_pipeline(args):
    from diffusers import QwenImageEditPipeline

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not visible; the full Qwen-Image-Edit model is not intended for CPU sampling"
        )
    pipe = QwenImageEditPipeline.from_pretrained(
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
    # Diffusers 0.35.1 unconditionally passes ``enable_gqa=False`` to SDPA,
    # but PyTorch 2.4 (the isolated environment's pinned version) does not
    # accept that keyword. Qwen has equal Q/KV image-attention head counts, so
    # the direct processor is numerically equivalent and works on 2.4.
    install_qwen_sdpa_compatibility(pipe)
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
            token_scope=args.token_scope,
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
        token_scope=args.token_scope,
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
    size = "dynamic1MP" if args.height is None else f"{args.height}x{args.width}"
    base = (
        f"Qwen-Image-Edit-{args.tore_type}-{size}-steps{args.num_inference_steps}-"
        f"cfg{args.true_cfg_scale:g}-seed{args.seed}-scope{args.token_scope}"
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


def pipeline_call_kwargs(
    args,
    prompts: Sequence[str],
    input_images: Sequence[Image.Image],
    generators: Sequence[torch.Generator],
    target_height: int,
    target_width: int,
) -> Dict[str, Any]:
    return {
        "image": list(input_images),
        "prompt": list(prompts),
        "negative_prompt": [args.negative_prompt] * len(prompts),
        "true_cfg_scale": args.true_cfg_scale,
        "generator": list(generators),
        "height": target_height,
        "width": target_width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "max_sequence_length": args.max_sequence_length,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
    temporary.replace(path)


def _atomic_save(image: Image.Image, path: Path) -> None:
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    image.save(temporary, quality=95)
    temporary.replace(path)


def _package_versions() -> Dict[str, str]:
    result = {"python": platform.python_version(), "torch": torch.__version__}
    for package in ("diffusers", "transformers", "accelerate"):
        try:
            module = __import__(package)
            result[package] = module.__version__
        except (ImportError, AttributeError):
            pass
    return result


def prepare_records(args, captions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    directory = Path(args.input_image_dir).expanduser().resolve()
    records = []
    for item in captions:
        path = directory / image_filename(item["image_id"])
        if not path.is_file():
            raise FileNotFoundError(f"missing COCO input image: {path}")
        bucket = dimension_bucket(path, args.height, args.width)
        records.append({**item, "input_path": path, "bucket": bucket})
    return records


def run(args) -> Path:
    captions = load_captions(args.caption_path)
    if args.limit is not None:
        captions = captions[: args.limit]
    records = prepare_records(args, captions)
    output_dir = Path(args.output_path).expanduser().resolve() / experiment_name(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    pending, skipped = [], 0
    for record in records:
        if (
            args.skip_existing
            and (output_dir / image_filename(record["image_id"])).is_file()
        ):
            skipped += 1
        else:
            pending.append(record)
    buckets: "OrderedDict[Tuple[int, int, int, int], List[Dict[str, Any]]]" = (
        OrderedDict()
    )
    for record in pending:
        buckets.setdefault(record["bucket"], []).append(record)

    if not pending:
        manifest_path = output_dir / "manifest.json"
        if manifest_path.is_file():
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        else:
            manifest = {
                "mode": args.tore_type,
                "model_path": str(Path(args.model_path).expanduser().resolve()),
                "parameters": vars(args),
                "versions": _package_versions(),
            }
        manifest.update(
            generated_this_run=0,
            skipped_existing=skipped,
            missing_count=0,
            resume_completed_without_model_load=True,
        )
        _write_json(manifest_path, manifest)
        print(
            f"All {len(records)} expected images already exist in {output_dir}; model loading skipped."
        )
        return output_dir

    pipe = apply_mode(load_pipeline(args), args)
    print(f"Mode: {args.tore_type}; patch: {patch_statistics(pipe)}")
    print(f"Output: {output_dir}; expected: {len(records)}; pending: {len(pending)}")

    if args.warmup_runs and records:
        record = records[0]
        target_height, target_width, _, _ = record["bucket"]
        with Image.open(record["input_path"]) as opened:
            warmup_image = opened.convert("RGB")
        seed = image_seed(args.seed, record["image_id"])
        for _ in range(args.warmup_runs):
            set_batch_seeds(pipe, [seed])
            generator = torch.Generator("cpu").manual_seed(seed)
            with torch.inference_mode():
                pipe(
                    **pipeline_call_kwargs(
                        args,
                        [record["caption"]],
                        [warmup_image],
                        [generator],
                        target_height,
                        target_width,
                    )
                )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    generated, started = 0, time.perf_counter()
    pipeline_invocations = transformer_forward_calls = 0
    execution_kinds: Dict[str, int] = {}
    minimum_reduced_tokens: int | None = None
    progress = tqdm(total=len(pending), desc="Generating edited images")
    for bucket, group in buckets.items():
        target_height, target_width, _, _ = bucket
        for start in range(0, len(group), args.batch_size):
            batch = group[start : start + args.batch_size]
            images = []
            for item in batch:
                with Image.open(item["input_path"]) as opened:
                    images.append(opened.convert("RGB"))
            seeds = [image_seed(args.seed, item["image_id"]) for item in batch]
            generators = [torch.Generator("cpu").manual_seed(seed) for seed in seeds]
            set_batch_seeds(pipe, seeds)
            with torch.inference_mode():
                output_images = pipe(
                    **pipeline_call_kwargs(
                        args,
                        [item["caption"] for item in batch],
                        images,
                        generators,
                        target_height,
                        target_width,
                    )
                ).images
            pipeline_invocations += 1
            current_stats = patch_statistics(pipe)
            calls_per_invocation = args.num_inference_steps * (
                2 if args.true_cfg_scale > 1 and args.negative_prompt is not None else 1
            )
            transformer_forward_calls += int(
                current_stats.get("transformer_calls", calls_per_invocation)
            )
            for call in current_stats.get("call_records", []):
                kind = str(call.get("kind", "full"))
                execution_kinds[kind] = execution_kinds.get(kind, 0) + 1
                reduced = int(
                    call.get("minimum_reduced_tokens", call.get("original_tokens", 0))
                )
                minimum_reduced_tokens = (
                    reduced
                    if minimum_reduced_tokens is None
                    else min(minimum_reduced_tokens, reduced)
                )
            for item, image in zip(batch, output_images):
                _atomic_save(image, output_dir / image_filename(item["image_id"]))
                generated += 1
                progress.update(1)
    progress.close()

    elapsed = time.perf_counter() - started
    expected = [image_filename(item["image_id"]) for item in records]
    missing = [name for name in expected if not (output_dir / name).is_file()]
    dimensions = {
        image_filename(item["image_id"]): {
            "target_height": item["bucket"][0],
            "target_width": item["bucket"][1],
            "reference_height": item["bucket"][2],
            "reference_width": item["bucket"][3],
        }
        for item in records
    }
    manifest = {
        "mode": args.tore_type,
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "input_image_dir": str(Path(args.input_image_dir).expanduser().resolve()),
        "caption_path": str(Path(args.caption_path).expanduser().resolve()),
        "parameters": vars(args),
        "token_scope": args.token_scope,
        "true_cfg_calls_per_step": 2
        if args.true_cfg_scale > 1 and args.negative_prompt is not None
        else 1,
        "cfg_transformer_calls_per_edited_image": args.num_inference_steps
        * (2 if args.true_cfg_scale > 1 and args.negative_prompt is not None else 1),
        "pipeline_invocations_this_run": pipeline_invocations,
        "transformer_forward_calls_this_run": transformer_forward_calls,
        "execution_kind_calls_this_run": execution_kinds,
        "minimum_reduced_image_tokens_this_run": minimum_reduced_tokens,
        "patch": patch_statistics(pipe),
        "dimensions_by_image": dimensions,
        "dimension_bucket_count": len({item["bucket"] for item in records}),
        "caption_count": len(records),
        "generated_this_run": generated,
        "skipped_existing": skipped,
        "missing_count": len(missing),
        "missing_files": missing[:100],
        "elapsed_seconds": elapsed,
        "seconds_per_generated_image": elapsed / max(generated, 1),
        "versions": _package_versions(),
    }
    _write_json(output_dir / "manifest.json", manifest)
    if missing:
        raise RuntimeError(
            f"generation finished with {len(missing)} missing images; see manifest.json"
        )
    print(
        f"Completed {len(records)} images in {elapsed:.2f}s ({generated} newly generated)."
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--caption-path",
        "--caption_path",
        dest="caption_path",
        default=str(DEFAULT_CAPTIONS),
    )
    parser.add_argument(
        "--input-image-dir",
        "--input_image_dir",
        dest="input_image_dir",
        default=str(DEFAULT_INPUT_IMAGES),
    )
    parser.add_argument(
        "--output-path",
        "--output_path",
        dest="output_path",
        default=str(DEFAULT_OUTPUT),
    )
    parser.add_argument(
        "--experiment-name", "--experiment_name", dest="experiment_name", default=None
    )
    parser.add_argument(
        "--model-path", "--model_path", dest="model_path", default=str(DEFAULT_MODEL)
    )
    parser.add_argument(
        "--torch-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--cpu-offload", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument(
        "--num-inference-steps",
        "--num_inference_steps",
        dest="num_inference_steps",
        type=int,
        default=50,
    )
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--true-cfg-scale",
        "--true_cfg_scale",
        dest="true_cfg_scale",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--negative-prompt", "--negative_prompt", dest="negative_prompt", default=" "
    )
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument(
        "--batch-size", "--batch_size", dest="batch_size", type=int, default=1
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tore-type", "--tore_type", dest="tore_type", choices=MODES, default="SDTM"
    )
    parser.add_argument(
        "--token-scope", choices=("target", "all-separate"), default="target"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument(
        "--show-pipeline-progress", action=argparse.BooleanOptionalAction, default=False
    )

    parser.add_argument("--ToMe-ratio", type=float, default=0.9)
    parser.add_argument("--ToMe-sx", type=int, default=2)
    parser.add_argument("--ToMe-sy", type=int, default=2)
    parser.add_argument(
        "--ToMe-use-rand", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--ToMe-merge-attn", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--ToMe-merge-mlp", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--ToMe-pseudo-merge", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument("--SDTM-ratio", type=float, default=0.3)
    parser.add_argument("--SDTM-deviation", type=float, default=0.2)
    parser.add_argument("--SDTM-switch-step", type=int, default=20)
    parser.add_argument(
        "--SDTM-use-rand", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--SDTM-sx", type=int, default=4)
    parser.add_argument("--SDTM-sy", type=int, default=4)
    parser.add_argument("--SDTM-a-s", type=float, default=0.05)
    parser.add_argument("--SDTM-a-d", type=float, default=0.05)
    parser.add_argument("--SDTM-a-p", type=float, default=2.0)
    parser.add_argument(
        "--SDTM-adaptive-ssm", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--SDTM-ssm-threshold", type=float, default=0.0)
    parser.add_argument("--SDTM-low-complexity-threshold", type=float, default=0.66)
    parser.add_argument("--SDTM-high-complexity-threshold", type=float, default=0.33)
    parser.add_argument(
        "--SDTM-pseudo-merge", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--SDTM-mcw", type=float, default=0.1)
    parser.add_argument("--SDTM-protect-steps-frequency", type=int, default=3)
    parser.add_argument("--SDTM-protect-layers-frequency", type=int, default=-1)
    parser.add_argument(
        "--SDTM-cache-each-step",
        dest="SDTM_cache_each_step",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--SDTM-slic-alpha", type=float, default=0.5)

    parser.add_argument("--Taylor-interval", type=int, default=2)
    parser.add_argument("--Taylor-max-order", type=int, default=1)
    parser.add_argument("--Taylor-first-enhance", type=int, default=12)
    return parser


def validate_args(args) -> None:
    if args.batch_size <= 0 or args.num_inference_steps <= 0:
        raise ValueError("batch size and inference steps must be positive")
    if (args.height is None) != (args.width is None):
        raise ValueError("--height and --width must be provided together")
    if args.height is not None and (
        args.height <= 0 or args.width <= 0 or args.height % 16 or args.width % 16
    ):
        raise ValueError("Qwen output dimensions must be positive multiples of 16")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.true_cfg_scale <= 0:
        raise ValueError("--true-cfg-scale must be positive")
    if args.tore_type == "ToMe":
        if not 0 <= args.ToMe_ratio <= 1 or args.ToMe_sx <= 0 or args.ToMe_sy <= 0:
            raise ValueError("invalid ToMe ratio/window")
    if args.tore_type.startswith("SDTM"):
        if not 0 <= args.SDTM_ratio <= 1:
            raise ValueError("--SDTM-ratio must be in [0, 1]")
        if (
            args.SDTM_deviation < 0
            or not 0 <= args.SDTM_ratio - args.SDTM_deviation <= 1
            or not 0 <= args.SDTM_ratio + args.SDTM_deviation <= 1
        ):
            raise ValueError(
                "the complete SDTM ratio +/- deviation schedule must remain in [0, 1]"
            )
        if (
            args.SDTM_switch_step < 0
            or args.SDTM_switch_step >= args.num_inference_steps
        ):
            raise ValueError("--SDTM-switch-step must identify an inference step")
        if args.SDTM_sx <= 0 or args.SDTM_sy <= 0:
            raise ValueError("SDTM window dimensions must be positive")
        if (
            args.tore_type in {"SDTM_testidm2", "SDTM_TaylorSeer"}
            and args.SDTM_sx * args.SDTM_sy < 4
        ):
            raise ValueError(
                "multi-level SSM requires at least four tokens per full window"
            )
    if (
        args.Taylor_interval <= 0
        or args.Taylor_max_order < 0
        or args.Taylor_first_enhance < 0
    ):
        raise ValueError("invalid TaylorSeer schedule")


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    validate_args(args)
    run(args)


if __name__ == "__main__":
    main()
