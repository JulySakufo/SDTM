"""Compute offline clean-fid scores for named FLUX image sets."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from evaluation_utils import parse_named_paths, validate_image_directory, write_json


HERE = Path(__file__).resolve().parent
DEFAULT_REAL_DIR = (HERE / "../datasets/val2017_center_crop_1024").resolve()
DEFAULT_INCEPTION_DIR = (HERE / "../inceptions").resolve()
INCEPTION_FILENAME = "inception-2015-12-05.pt"


class OfflineInception(torch.nn.Module):
    def __init__(self, model_directory: str, device: str):
        super().__init__()
        from cleanfid.inception_torchscript import InceptionV3W

        self.model = InceptionV3W(model_directory, download=False).to(device)

    def forward(self, value):
        return self.model(value)


def extract_fid_features(paths, extractor, *, device: str, batch_size: int, description: str):
    from cleanfid import fid

    if len(paths) < 2:
        raise ValueError("FID requires at least two images per set")
    return fid.get_files_features(
        paths, model=extractor, device=device, batch_size=batch_size, mode="clean", description=description
    )


def compute_fid_score(reference_features, generated_features) -> float:
    from cleanfid import fid

    return float(fid.fid_from_feats(reference_features, generated_features))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", default=str(DEFAULT_REAL_DIR))
    parser.add_argument("--image-set", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--inception-dir", default=str(DEFAULT_INCEPTION_DIR))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--expected-count", type=int, default=5000)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output-json", default="fid_scores_flux.json")
    args = parser.parse_args()

    from cleanfid import fid

    real = Path(args.real_dir).expanduser().resolve()
    real_validation = validate_image_directory(real, expected_count=args.expected_count)
    if not real_validation["valid"] and not args.allow_incomplete:
        raise RuntimeError(f"real image set failed validation: {real_validation}")
    extractor = OfflineInception(args.inception_dir, args.device)
    results = {}
    for name, directory in parse_named_paths(args.image_set).items():
        validation = validate_image_directory(directory, expected_count=args.expected_count)
        if not validation["valid"] and not args.allow_incomplete:
            raise RuntimeError(f"{name} failed validation: {validation}")
        score = fid.compute_fid(str(real), str(directory), custom_feat_extractor=extractor, device=args.device)
        results[name] = {"fid": float(score), "validation": validation}
        print(f"{name}: FID {score:.4f}")
    write_json(
        args.output_json,
        {"real_validation": real_validation, "inception_dir": args.inception_dir, "results": results},
    )


if __name__ == "__main__":
    main()
