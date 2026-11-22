"""Compute offline OpenCLIP ViT-G/14 scores for named FLUX image sets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from evaluation_utils import list_images, parse_named_paths, validate_image_directory, write_json


HERE = Path(__file__).resolve().parent
DEFAULT_CLIP_MODEL = (HERE / "../clip_models/ViT-G-14.pt").resolve()
DEFAULT_CLIP_MODEL_NAME = "ViT-g-14"


def load_clip_model(checkpoint: str | Path, model_name: str, device: str):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    del state
    return model.to(device).eval(), preprocess, open_clip.get_tokenizer(model_name)


def compute_clip_score(
    samples: Sequence[tuple[Path, str]], clip_model, *, batch_size: int, device: str, description: str
) -> float:
    model, preprocess, tokenizer = clip_model
    scores = []
    device_type = torch.device(device).type
    for start in tqdm(range(0, len(samples), batch_size), desc=description):
        batch = samples[start : start + batch_size]
        images = []
        for path, _ in batch:
            with Image.open(path) as image:
                images.append(preprocess(image.convert("RGB")))
        image_inputs = torch.stack(images).to(device)
        text_inputs = tokenizer([caption for _, caption in batch]).to(device)
        with torch.inference_mode(), torch.autocast(device_type=device_type, enabled=device_type == "cuda"):
            image_features = F_normalize(model.encode_image(image_inputs))
            text_features = F_normalize(model.encode_text(text_inputs))
            batch_scores = (100.0 * (image_features * text_features).sum(-1)).clamp_min(0)
        scores.extend(batch_scores.float().cpu().tolist())
    return float(np.mean(scores))


def main() -> None:
    from sample import DEFAULT_CAPTIONS, image_filename, load_captions

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", default=str(DEFAULT_CAPTIONS))
    parser.add_argument("--image-set", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--clip-model", default=str(DEFAULT_CLIP_MODEL))
    parser.add_argument("--model-name", default=DEFAULT_CLIP_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--expected-count", type=int, default=5000)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output-json", default="clip_scores_flux.json")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    captions = load_captions(args.captions)
    caption_by_stem = {Path(image_filename(item["image_id"])).stem: item["caption"] for item in captions}
    expected_stems = list(caption_by_stem)
    clip_model = load_clip_model(args.clip_model, args.model_name, args.device)

    results = {}
    for name, directory in parse_named_paths(args.image_set).items():
        validation = validate_image_directory(
            directory, expected_count=args.expected_count, expected_stems=expected_stems
        )
        if not validation["valid"] and not args.allow_incomplete:
            raise RuntimeError(f"{name} failed validation: {validation}")
        samples = [(path, caption_by_stem[path.stem]) for path in list_images(directory) if path.stem in caption_by_stem]
        score = compute_clip_score(
            samples, clip_model, batch_size=args.batch_size, device=args.device, description=f"CLIP {name}"
        ) if samples else None
        results[name] = {
            "clip_score": score,
            "evaluated_images": len(samples),
            "validation": validation,
        }
        print(f"{name}: {score:.4f} ({len(samples)} images)" if score is not None else f"{name}: no matching images")
    write_json(args.output_json, {"model": args.model_name, "results": results})


def F_normalize(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)


if __name__ == "__main__":
    main()
