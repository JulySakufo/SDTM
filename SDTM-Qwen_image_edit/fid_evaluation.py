"""Compute clean-fid against COCO val2017 center-crop references."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from evaluation_utils import parse_named_paths, validate_image_directory, write_json
from sample import DEFAULT_CAPTIONS, image_filename, load_captions


HERE = Path(__file__).resolve().parent


class OfflineInception(torch.nn.Module):
    def __init__(self, model_directory: str, device: str):
        super().__init__()
        from cleanfid.inception_torchscript import InceptionV3W

        self.model = InceptionV3W(model_directory, download=False).to(device)

    def forward(self, value):
        return self.model(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", default=str(DEFAULT_CAPTIONS))
    parser.add_argument(
        "--real-dir",
        default=str((HERE / "../datasets/val2017_center_crop_1024").resolve()),
    )
    parser.add_argument(
        "--image-set", action="append", required=True, metavar="NAME=PATH"
    )
    parser.add_argument(
        "--inception-dir", default=str((HERE / "../inceptions").resolve())
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--expected-count", type=int, default=5000)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output-json", default="fid_scores_qwen_image_edit.json")
    args = parser.parse_args()

    from cleanfid import fid

    expected_stems = [
        Path(image_filename(item["image_id"])).stem
        for item in load_captions(args.captions)
    ]
    real = Path(args.real_dir).expanduser().resolve()
    real_validation = validate_image_directory(
        real,
        expected_count=args.expected_count,
        expected_stems=expected_stems,
    )
    if not real_validation["valid"] and not args.allow_incomplete:
        raise RuntimeError(f"real image set failed validation: {real_validation}")
    extractor = OfflineInception(args.inception_dir, args.device)
    results = {}
    for name, directory in parse_named_paths(args.image_set).items():
        validation = validate_image_directory(
            directory,
            expected_count=args.expected_count,
            expected_stems=expected_stems,
        )
        if not validation["valid"] and not args.allow_incomplete:
            raise RuntimeError(f"{name} failed validation: {validation}")
        score = fid.compute_fid(
            str(real), str(directory), custom_feat_extractor=extractor
        )
        results[name] = {"fid": float(score), "validation": validation}
        print(f"{name}: FID {score:.4f}")
    write_json(
        args.output_json,
        {
            "real_validation": real_validation,
            "inception_dir": args.inception_dir,
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
