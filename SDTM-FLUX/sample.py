"""Generate and benchmark FLUX.1-dev with a caption dataset or custom prompts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from tqdm import tqdm

from clip_evaluation import DEFAULT_CLIP_MODEL, DEFAULT_CLIP_MODEL_NAME, compute_clip_score, load_clip_model
from evaluation_utils import matching_images, write_json
from fid_evaluation import (
    DEFAULT_INCEPTION_DIR,
    DEFAULT_REAL_DIR,
    INCEPTION_FILENAME,
    OfflineInception,
    compute_fid_score,
    extract_fid_features,
)
from flop_profile import profile_transformer
from flux_core import patch_statistics, set_batch_seeds


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = (HERE / "../../models/FLUX.1-dev").resolve()
DEFAULT_CAPTIONS = (HERE / "../datasets/COCO2017/captions_val2017.json").resolve()
MODES = ("Default", "ToMe", "SDTM", "SDTM_testidm2", "SDTM_TaylorSeer")


def parse_modes(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(MODES)
    aliases = {mode.lower(): mode for mode in MODES}
    aliases.update(baseline="Default", testidm2="SDTM_testidm2", taylorseer="SDTM_TaylorSeer")
    modes = []
    for token in filter(None, (item.strip() for item in value.split(","))):
        mode = aliases.get(token.lower())
        if mode is None:
            raise ValueError(f"unknown mode {token!r}; choose from {', '.join(MODES)}")
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise ValueError("at least one method is required")
    return modes


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
    if not value or value in {".", ".."} or any(char in value for char in "/\\\0"):
        raise ValueError(f"invalid image ID: {image_id!r}")
    return f"{int(value):012d}.jpg" if value.isdecimal() else f"{value}.jpg"


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


def batch_call_kwargs(pipe, args, batch: Sequence[Dict[str, Any]]) -> dict:
    seeds = [image_seed(args.seed, item["image_id"]) for item in batch]
    set_batch_seeds(pipe, seeds)
    generators = [torch.Generator("cpu").manual_seed(seed) for seed in seeds]
    return pipeline_call_kwargs(args, [item["caption"] for item in batch], generators)


def timed_generation(pipe, args, batch):
    kwargs = batch_call_kwargs(pipe, args, batch)
    cuda_device = torch.device(args.device) if args.device.startswith("cuda") else None
    with torch.inference_mode():
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
        started = time.perf_counter()
        images = pipe(**kwargs).images
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
        elapsed = time.perf_counter() - started
    if len(images) != len(batch):
        raise RuntimeError(f"expected {len(batch)} generated images, received {len(images)}")
    return images, elapsed


def generate_images(args, captions, output_dir: Path) -> dict:
    pipe = apply_mode(load_pipeline(args), args)
    try:
        first_batch = captions[: args.batch_size]
        for _ in range(args.warmup_runs):
            with torch.inference_mode():
                pipe(**batch_call_kwargs(pipe, args, first_batch))

        generated = skipped = 0
        elapsed = 0.0
        for start in tqdm(range(0, len(captions), args.batch_size), desc=f"Generating {args.tore_type}"):
            batch = captions[start : start + args.batch_size]
            if args.skip_existing:
                pending = [item for item in batch if not (output_dir / image_filename(item["image_id"])).is_file()]
                skipped += len(batch) - len(pending)
                batch = pending
            if not batch:
                continue
            images, duration = timed_generation(pipe, args, batch)
            elapsed += duration
            for item, image in zip(batch, images):
                destination = output_dir / image_filename(item["image_id"])
                temporary = destination.with_suffix(".jpg.tmp")
                image.save(temporary, format="JPEG", quality=95)
                temporary.replace(destination)
            generated += len(batch)

        timed_images = generated
        if not timed_images:
            _, elapsed = timed_generation(pipe, args, first_batch)
            timed_images = len(first_batch)

        flops = profile_transformer(pipe, args.tore_type, batch_call_kwargs(pipe, args, first_batch))
        flops["profiled_image_ids"] = [item["image_id"] for item in first_batch]
        return {
            "generated_this_run": generated,
            "skipped_existing": skipped,
            "timed_images": timed_images,
            "elapsed_seconds": elapsed,
            "seconds_per_image": elapsed / timed_images,
            "transformer_parameters": sum(parameter.numel() for parameter in pipe.transformer.parameters()),
            "transformer_flops_per_image": flops["total_flops_per_image"],
            "patch": patch_statistics(pipe),
            "flops": flops,
        }
    finally:
        del pipe
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


def evaluate_dataset(args, captions, generated_paths, reference_paths) -> dict:
    device = args.metric_device or args.device
    clip_model = load_clip_model(Path(args.clip_model).expanduser(), args.clip_model_name, device)
    samples = [(path, item["caption"]) for path, item in zip(generated_paths, captions)]
    clip_score = compute_clip_score(
        samples, clip_model, device=device, batch_size=args.metric_batch_size, description=f"CLIP {args.tore_type}"
    )
    del clip_model

    extractor = OfflineInception(str(Path(args.inception_dir).expanduser()), device)
    feature_options = dict(device=device, batch_size=args.metric_batch_size)
    references = extract_fid_features(reference_paths, extractor, description="FID references", **feature_options)
    generated = extract_fid_features(generated_paths, extractor, description=f"FID {args.tore_type}", **feature_options)
    return {"clip_score": clip_score, "fid": compute_fid_score(references, generated), "evaluated_images": len(samples)}


def run(args) -> Path:
    name = experiment_name(args)
    if args.prompt is not None:
        if not args.prompt or any(not prompt.strip() for prompt in args.prompt):
            raise ValueError("--prompt must contain non-whitespace text")
        captions = [{"image_id": index, "caption": prompt} for index, prompt in enumerate(args.prompt)]
        source = {"type": "prompts"}
        use_input_suffix = True
    else:
        caption_path = Path(args.caption_path or DEFAULT_CAPTIONS).expanduser().resolve()
        captions = load_captions(caption_path)
        source = {"type": "dataset", "caption_path": str(caption_path)}
        use_input_suffix = caption_path != DEFAULT_CAPTIONS
    if not args.experiment_name and use_input_suffix:
        digest = hashlib.sha256(json.dumps(captions, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
        name += f"-{source['type']}-{digest}"

    captions = captions[: args.limit]
    if not captions:
        raise ValueError("the selected input contains no prompts")
    filenames = [image_filename(item["image_id"]) for item in captions]
    if len(set(filenames)) != len(filenames):
        raise ValueError("image IDs must map to unique output filenames")
    output_dir = Path(args.output_path).expanduser().resolve() / name
    generated_paths = [output_dir / filename for filename in filenames]
    reference_paths = None
    if source["type"] == "dataset":
        if len(captions) < 2:
            raise ValueError("dataset evaluation requires at least two images to compute FID")
        real_dir = Path(args.real_dir).expanduser().resolve()
        if real_dir == output_dir.resolve():
            raise ValueError("the reference and output directories must be different")
        reference_paths = matching_images(real_dir, filenames)
        source["real_dir"] = str(real_dir)
        metric_weights = (Path(args.clip_model).expanduser(), Path(args.inception_dir).expanduser() / INCEPTION_FILENAME)
        for weights in metric_weights:
            if not weights.is_file():
                raise FileNotFoundError(f"dataset metric checkpoint not found: {weights}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.tore_type} will write {len(captions)} images to {output_dir}")
    performance = generate_images(args, captions, output_dir)
    missing = [path.name for path in generated_paths if not path.is_file()]
    manifest = {
        "mode": args.tore_type,
        "model_path": str(Path(args.model_path).resolve()),
        "input": source,
        "parameters": vars(args),
        "caption_count": len(captions),
        "missing_count": len(missing),
        "missing_files": missing[:100],
        "torch_version": torch.__version__,
        **performance,
    }
    try:
        import diffusers

        manifest["diffusers_version"] = diffusers.__version__
    except ImportError:
        pass
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(
        f"{args.tore_type} | {manifest['seconds_per_image']:.4f} s/image "
        f"({manifest['timed_images']} timed) | {manifest['transformer_parameters']:,} transformer parameters "
        f"| {manifest['transformer_flops_per_image'] / 1e12:.5g} TFLOPs/image"
    )
    if missing:
        raise RuntimeError(f"generation finished with {len(missing)} missing images; see manifest.json")
    if reference_paths is not None:
        manifest.update(evaluate_dataset(args, captions, generated_paths, reference_paths))
        write_json(manifest_path, manifest)
        print(f"{args.tore_type} | CLIP Score {manifest['clip_score']:.4f} | FID {manifest['fid']:.4f}")
    print(f"Results saved to {manifest_path}")
    return output_dir


def build_parser(default_mode: str | None = "SDTM") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--dataset", "--caption-path", dest="caption_path", metavar="JSON",
                        help="COCO-style caption JSON (defaults to the local COCO 2017 validation captions)")
    inputs.add_argument("--prompt", action="append", help="repeat this option to supply multiple custom prompts")
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
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--tore-type", choices=MODES, default=default_mode, help="run a single method")
    modes.add_argument("--modes", help="comma-separated methods or 'all'")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--show-pipeline-progress", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--real-dir", default=str(DEFAULT_REAL_DIR), help="reference images for dataset FID")
    parser.add_argument("--clip-model", default=str(DEFAULT_CLIP_MODEL))
    parser.add_argument("--clip-model-name", default=DEFAULT_CLIP_MODEL_NAME)
    parser.add_argument("--inception-dir", default=str(DEFAULT_INCEPTION_DIR))
    parser.add_argument("--metric-batch-size", type=int, default=32)
    parser.add_argument("--metric-device", help="device for dataset metrics (defaults to --device)")

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
    if args.warmup_runs < 0 or args.metric_batch_size <= 0:
        raise ValueError("warmup runs must be non-negative and metric batch size must be positive")
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


def main(argv: Iterable[str] | None = None, *, default_mode: str = "SDTM") -> None:
    parser = build_parser(default_mode)
    args = parser.parse_args(argv)
    try:
        modes = [args.tore_type] if args.modes is None else parse_modes(args.modes)
        configurations = []
        for mode in modes:
            config = argparse.Namespace(**vars(args))
            config.tore_type = mode
            if args.experiment_name and len(modes) > 1:
                config.experiment_name = f"{args.experiment_name}-{mode}"
            validate_args(config)
            configurations.append(config)
    except ValueError as error:
        parser.error(str(error))
    for config in configurations:
        run(config)


if __name__ == "__main__":
    main()
