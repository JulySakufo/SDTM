"""Shared strict image-set validation and atomic JSON writing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_named_paths(values: Iterable[str]) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=PATH, received {value!r}")
        name, path = value.split("=", 1)
        if not name or name in result:
            raise ValueError(f"invalid or duplicate image-set name: {name!r}")
        result[name] = Path(path).expanduser().resolve()
    if not result:
        raise ValueError("at least one --image-set NAME=PATH is required")
    return result


def list_images(directory: Path) -> List[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {directory}")
    return sorted(
        path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )


def validate_image_directory(
    directory: Path,
    expected_count: int | None = None,
    expected_stems: Iterable[str] | None = None,
    verify: bool = True,
) -> Dict:
    images = list_images(directory)
    corrupt = []
    if verify:
        for path in images:
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception:
                corrupt.append(path.name)
    stems = {path.stem for path in images}
    expected = set(expected_stems or [])
    missing = sorted(expected - stems)
    extra = sorted(stems - expected) if expected else []
    return {
        "directory": str(directory),
        "image_count": len(images),
        "expected_count": expected_count,
        "missing_count": len(missing),
        "missing": missing[:100],
        "extra_count": len(extra),
        "extra": extra[:100],
        "corrupt_count": len(corrupt),
        "corrupt": corrupt[:100],
        "valid": not corrupt
        and not missing
        and (expected_count is None or len(images) == expected_count),
    }


def write_json(path: str | Path, value) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
    temporary.replace(destination)
