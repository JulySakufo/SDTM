"""Compute offline OpenCLIP ViT-G/14 scores for named FLUX image sets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from evaluation_utils import parse_named_paths, validate_image_directory, write_json
from sample import DEFAULT_CAPTIONS, image_filename, load_captions


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", default=str(DEFAULT_CAPTIONS))
    parser.add_argument("--image-set", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--clip-model", default=str((HERE / "../clip_models/ViT-G-14.pt").resolve()))
    parser.add_argument("--model-name", default="ViT-g-14")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--expected-count", type=int, default=5000)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output-json", default="clip_scores_flux.json")
    args = parser.parse_args()

    import open_clip

    captions = load_captions(args.captions)
    caption_by_stem = {Path(image_filename(item["image_id"])).stem: item["caption"] for item in captions}
    expected_stems = list(caption_by_stem)
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name)
    state = torch.load(args.clip_model, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model = model.to(args.device).eval()
    tokenizer = open_clip.get_tokenizer(args.model_name)

    results = {}
    for name, directory in parse_named_paths(args.image_set).items():
        validation = validate_image_directory(
            directory, expected_count=args.expected_count, expected_stems=expected_stems
        )
        if not validation["valid"] and not args.allow_incomplete:
            raise RuntimeError(f"{name} failed validation: {validation}")
        paths = [path for path in sorted(directory.iterdir()) if path.stem in caption_by_stem]
        scores = []
        for start in tqdm(range(0, len(paths), args.batch_size), desc=f"CLIP {name}"):
            batch_paths = paths[start : start + args.batch_size]
            images = torch.stack([preprocess(Image.open(path).convert("RGB")) for path in batch_paths]).to(args.device)
            texts = tokenizer([caption_by_stem[path.stem] for path in batch_paths]).to(args.device)
            with torch.inference_mode(), torch.autocast(device_type=args.device.split(":")[0], enabled=args.device.startswith("cuda")):
                image_features = F_normalize(model.encode_image(images))
                text_features = F_normalize(model.encode_text(texts))
                batch_scores = (100.0 * (image_features * text_features).sum(-1)).clamp_min(0)
            scores.extend(batch_scores.float().cpu().tolist())
        results[name] = {
            "clip_score": float(np.mean(scores)) if scores else None,
            "evaluated_images": len(scores),
            "validation": validation,
        }
        print(f"{name}: {results[name]['clip_score']:.4f} ({len(scores)} images)")
    write_json(args.output_json, {"model": args.model_name, "results": results})


def F_normalize(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)


if __name__ == "__main__":
    main()
