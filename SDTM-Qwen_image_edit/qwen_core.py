"""Runtime-only SDTM/ToMe patches for Diffusers Qwen-Image-Edit.

The image stream is a concatenation of the target noisy latent and the encoded
reference image.  Plans are built per segment so a merge can never cross that
boundary.  Text tokens are never merged.  The implementation targets
Diffusers 0.35.1 and does not modify the installed package.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from taylor_cache import reset_taylor, taylor_cal_type, taylor_formula, update_taylor


Tensor = torch.Tensor


def _gather_tokens(x: Tensor, index: Tensor) -> Tensor:
    return torch.gather(x, 1, index.expand(x.shape[0], index.shape[1], x.shape[-1]))


@dataclass
class MergePlan:
    """A reversible mapping from a full segment to representative tokens."""

    num_tokens: int
    kept_idx: Tensor
    source_idx: Tensor
    source_to_reduced: Tensor
    destination_idx: Tensor
    destination_to_reduced: Tensor
    unmerged_idx: Tensor

    @property
    def batch_size(self) -> int:
        return self.kept_idx.shape[0]

    @property
    def reduced_tokens(self) -> int:
        return self.kept_idx.shape[1]

    def merge(self, x: Tensor, mode: str = "mean") -> Tensor:
        output = _gather_tokens(x, self.kept_idx)
        if self.source_idx.shape[1] == 0:
            return output
        source = _gather_tokens(x, self.source_idx)
        return output.scatter_reduce(
            1,
            self.source_to_reduced.expand(
                source.shape[0], source.shape[1], source.shape[-1]
            ),
            source,
            reduce=mode,
            include_self=True,
        )

    def prune(self, x: Tensor) -> Tensor:
        return _gather_tokens(x, self.kept_idx)

    def apply(self, x: Tensor, pseudo_merge: bool = False) -> Tensor:
        return self.prune(x) if pseudo_merge else self.merge(x)

    def unmerge(
        self, x: Tensor, cache: Optional[Tensor] = None, mcw: float = 1.0
    ) -> Tensor:
        batch, _, channels = x.shape
        kept_values = x
        source_values = _gather_tokens(x, self.source_to_reduced)
        if cache is not None:
            cache = cache.to(device=x.device, dtype=x.dtype)
            if self.destination_idx.shape[1]:
                destination_values = _gather_tokens(x, self.destination_to_reduced)
                cached_destination = _gather_tokens(cache, self.destination_idx)
                destination_values = (
                    mcw * destination_values + (1.0 - mcw) * cached_destination
                )
                kept_values = kept_values.clone()
                kept_values.scatter_(
                    1,
                    self.destination_to_reduced.expand(
                        batch, self.destination_to_reduced.shape[1], channels
                    ),
                    destination_values,
                )
            if self.source_idx.shape[1]:
                cached_source = _gather_tokens(cache, self.source_idx)
                source_values = mcw * source_values + (1.0 - mcw) * cached_source

        full = torch.zeros(
            batch, self.num_tokens, channels, device=x.device, dtype=x.dtype
        )
        full.scatter_(
            1, self.kept_idx.expand(batch, self.reduced_tokens, channels), kept_values
        )
        if self.source_idx.shape[1]:
            full.scatter_(
                1,
                self.source_idx.expand(batch, self.source_idx.shape[1], channels),
                source_values,
            )
        return full


@dataclass
class SegmentPlan:
    name: str
    length: int
    plan: Optional[MergePlan]

    @property
    def reduced_length(self) -> int:
        return self.length if self.plan is None else self.plan.reduced_tokens


@dataclass
class CompositeMergePlan:
    """Independent plans for target/reference segments of the image stream."""

    segments: List[SegmentPlan]
    batch_size: int

    @property
    def num_tokens(self) -> int:
        return sum(segment.length for segment in self.segments)

    @property
    def reduced_tokens(self) -> int:
        return sum(segment.reduced_length for segment in self.segments)

    def apply(self, x: Tensor, pseudo_merge: bool = False) -> Tensor:
        pieces, offset = [], 0
        for segment in self.segments:
            piece = x[:, offset : offset + segment.length]
            pieces.append(
                piece
                if segment.plan is None
                else segment.plan.apply(piece, pseudo_merge)
            )
            offset += segment.length
        return torch.cat(pieces, dim=1)

    def unmerge(
        self, x: Tensor, cache: Optional[Tensor] = None, mcw: float = 1.0
    ) -> Tensor:
        pieces, original_offset, reduced_offset = [], 0, 0
        for segment in self.segments:
            value = x[:, reduced_offset : reduced_offset + segment.reduced_length]
            cached = (
                None
                if cache is None
                else cache[:, original_offset : original_offset + segment.length]
            )
            pieces.append(
                value
                if segment.plan is None
                else segment.plan.unmerge(value, cached, mcw)
            )
            original_offset += segment.length
            reduced_offset += segment.reduced_length
        return torch.cat(pieces, dim=1)

    def gather_image_rope(self, frequencies: Tensor) -> Tensor:
        if frequencies.ndim == 2:
            frequencies = frequencies.unsqueeze(0).expand(self.batch_size, -1, -1)
        if frequencies.ndim != 3 or frequencies.shape[1] != self.num_tokens:
            raise ValueError(
                f"image RoPE must be [tokens, dim] or [batch, tokens, dim], got {tuple(frequencies.shape)}"
            )
        pieces, offset = [], 0
        for segment in self.segments:
            value = frequencies[:, offset : offset + segment.length]
            pieces.append(
                value
                if segment.plan is None
                else _gather_tokens(value, segment.plan.kept_idx)
            )
            offset += segment.length
        return torch.cat(pieces, dim=1)

    def global_kept_indices(self) -> Tensor:
        pieces, offset = [], 0
        device = next(
            (s.plan.kept_idx.device for s in self.segments if s.plan is not None),
            torch.device("cpu"),
        )
        for segment in self.segments:
            if segment.plan is None:
                local = (
                    torch.arange(segment.length, device=device)
                    .view(1, -1, 1)
                    .expand(self.batch_size, -1, -1)
                )
            else:
                local = segment.plan.kept_idx
            pieces.append(local + offset)
            offset += segment.length
        return torch.cat(pieces, dim=1)


def _empty_plan(x: Tensor) -> MergePlan:
    batch, tokens, _ = x.shape
    kept = (
        torch.arange(tokens, device=x.device).view(1, tokens, 1).expand(batch, -1, -1)
    )
    empty = torch.empty(batch, 0, 1, device=x.device, dtype=torch.long)
    return MergePlan(tokens, kept, empty, empty, empty, empty, kept)


def _random_scores(
    info: Dict, batch: int, length: int, device: torch.device, salt: int
) -> Tensor:
    if not bool(info["args"].get("use_rand", True)):
        return (
            torch.arange(length, device=device, dtype=torch.float32)
            .view(1, -1)
            .expand(batch, -1)
        )
    seeds = info["state"].get("sample_seeds")
    step = int(info["state"].get("step", 0))
    layer = int(info["state"].get("global_layer", 0))
    counter = int(info["state"].get("plan_counter", 0))
    info["state"]["plan_counter"] = counter + 1
    offset = step * 1_000_003 + layer * 10_007 + counter * 101 + salt
    if seeds is None:
        seeds = [int(info["args"].get("seed", 0)) + index for index in range(batch)]
    rows = []
    for index in range(batch):
        generator = torch.Generator(device=device)
        generator.manual_seed((int(seeds[index]) + offset) % (2**63 - 1))
        rows.append(torch.rand(length, device=device, generator=generator))
    return torch.stack(rows)


def _window_indices(
    height: int, width: int, sy: int, sx: int, device: torch.device
) -> Tuple[Tensor, Tensor]:
    """Return full windows plus border indices that must never be merged."""
    if sy <= 0 or sx <= 0:
        raise ValueError("merge window dimensions must be positive")
    grid = torch.arange(height * width, device=device).view(height, width)
    full_height, full_width = height // sy * sy, width // sx * sx
    if full_height and full_width:
        windows = (
            grid[:full_height, :full_width]
            .view(full_height // sy, sy, full_width // sx, sx)
            .permute(0, 2, 1, 3)
            .reshape(-1, sy * sx)
        )
    else:
        windows = torch.empty(0, sy * sx, dtype=torch.long, device=device)
    interior = torch.zeros(height, width, dtype=torch.bool, device=device)
    interior[:full_height, :full_width] = True
    border = grid[~interior].reshape(-1)
    return windows, border


def _plan_from_groups(
    num_tokens: int,
    unmerged: Tensor,
    destinations: Tensor,
    sources: Tensor,
    source_to_destination: Tensor,
) -> MergePlan:
    batch = unmerged.shape[0]
    kept = torch.cat([unmerged, destinations], dim=1).unsqueeze(-1)
    destination_offset = unmerged.shape[1]
    destination_to_reduced = (
        torch.arange(destinations.shape[1], device=destinations.device)
        .view(1, -1)
        .expand(batch, -1)
        + destination_offset
    ).unsqueeze(-1)
    return MergePlan(
        num_tokens,
        kept,
        sources.unsqueeze(-1),
        (source_to_destination + destination_offset).unsqueeze(-1),
        destinations.unsqueeze(-1),
        destination_to_reduced,
        unmerged.unsqueeze(-1),
    )


def _independence_key(info: Dict, segment: str) -> Tuple[str, str]:
    return str(info["state"].get("cfg_branch", "cond")), segment


def _get_independence(info: Dict, segment: str) -> Optional[Tensor]:
    return (
        info["state"]
        .setdefault("last_independent", {})
        .get(_independence_key(info, segment))
    )


def _update_independence(
    info: Dict, plan: MergePlan, pseudo_merge: bool, segment: str
) -> None:
    values = _get_independence(info, segment)
    if values is None or values.shape != (plan.batch_size, plan.num_tokens):
        values = torch.zeros(
            plan.batch_size,
            plan.num_tokens,
            device=plan.kept_idx.device,
            dtype=torch.int32,
        )
    values.add_(1)
    independent = (
        plan.kept_idx.squeeze(-1) if pseudo_merge else plan.unmerged_idx.squeeze(-1)
    )
    if independent.shape[1]:
        values.scatter_(
            1, independent, torch.zeros_like(independent, dtype=values.dtype)
        )
    info["state"].setdefault("last_independent", {})[
        _independence_key(info, segment)
    ] = values


def make_tome_plan(
    x: Tensor, height: int, width: int, info: Dict, segment: str = "target"
) -> MergePlan:
    batch, tokens, _ = x.shape
    ratio = float(info["args"].get("ratio", 0.9))
    remove = int(tokens * (1.0 - ratio))
    if tokens >= 256:
        remove = remove // 16 * 16
    windows, border = _window_indices(
        height, width, int(info["args"]["sy"]), int(info["args"]["sx"]), x.device
    )
    if remove <= 0 or windows.numel() == 0 or windows.shape[1] <= 1:
        return _empty_plan(x)
    batch_windows = windows.unsqueeze(0).expand(batch, -1, -1)
    order = _random_scores(
        info, batch, windows.numel(), x.device, 11 + _segment_salt(segment)
    )
    order = order.view(batch, *windows.shape).argsort(-1)
    shuffled = batch_windows.gather(2, order)
    destinations = shuffled[:, :, 0]
    candidates = shuffled[:, :, 1:].reshape(batch, -1)
    normalized = F.normalize(x.float(), dim=-1).to(x.dtype)
    scores = _gather_tokens(normalized, candidates.unsqueeze(-1)) @ _gather_tokens(
        normalized, destinations.unsqueeze(-1)
    ).transpose(1, 2)
    best_score, best_destination = scores.max(-1)
    remove = min(remove, candidates.shape[1])
    ranked = best_score.argsort(-1, descending=True)
    sources = candidates.gather(1, ranked[:, :remove])
    unmerged = candidates.gather(1, ranked[:, remove:])
    if border.numel():
        unmerged = torch.cat([unmerged, border.view(1, -1).expand(batch, -1)], dim=1)
    mapping = best_destination.gather(1, ranked[:, :remove])
    return _plan_from_groups(tokens, unmerged, destinations, sources, mapping)


def _segment_salt(segment: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(segment)) % 997


def _window_statistics(
    x: Tensor, windows: Tensor, info: Dict, improved: bool, segment: str
) -> Tuple[Tensor, Tensor, Tensor]:
    batch, _, channels = x.shape
    window_count, size = windows.shape
    expanded = windows.reshape(1, -1, 1).expand(batch, -1, -1)
    values = _gather_tokens(x, expanded).view(batch, window_count, size, channels)
    normalized = F.normalize(values.float(), dim=-1)
    similarities = normalized @ normalized.transpose(-1, -2)
    similarity_score = similarities.mean(dim=(-1, -2))
    variance_raw = similarities.var(dim=(-1, -2), unbiased=False)
    minimum, maximum = (
        variance_raw.amin(1, keepdim=True),
        variance_raw.amax(1, keepdim=True),
    )
    variance_score = 1.0 - (variance_raw - minimum) / (maximum - minimum + 1e-6)
    priority = torch.zeros_like(similarity_score)
    last = _get_independence(info, segment)
    if last is not None and last.shape[1] == x.shape[1]:
        normalized_priority = last.float() / (last.float().mean(1, keepdim=True) + 1e-6)
        priority = (
            _gather_tokens(normalized_priority.unsqueeze(-1), expanded)
            .view(batch, window_count, size)
            .mean(-1)
        )
    score = similarity_score + float(info["args"].get("a_s", 0.05)) * priority
    if improved:
        score = score + variance_score
    return score, variance_score, normalized.to(x.dtype)


def make_ssm_plan(
    x: Tensor,
    height: int,
    width: int,
    info: Dict,
    variant: str,
    segment: str = "target",
) -> MergePlan:
    batch, tokens, _ = x.shape
    args = info["args"]
    windows, border = _window_indices(
        height, width, int(args["sy"]), int(args["sx"]), x.device
    )
    if windows.numel() == 0 or windows.shape[1] <= 1:
        return _empty_plan(x)
    window_count, size = windows.shape
    improved = variant in {"testidm2", "legacy_taylor"}
    target = max(0, int(tokens * float(info["state"].get("ratio_current", 0.0))))
    if target <= 0 and not bool(args.get("adaptive_ssm", False)):
        return _empty_plan(x)
    score, variance_score, normalized_windows = _window_statistics(
        x, windows, info, improved, segment
    )

    if not improved:
        count = min(window_count, target // (size - 1))
        if count <= 0:
            return _empty_plan(x)
        keep_counts = [(score.topk(count, dim=-1).indices, 1)]
    else:
        if size < 4:
            raise ValueError(
                "SDTM_testidm2 requires at least four tokens in a full merge window"
            )
        if bool(args.get("adaptive_ssm", False)):
            low = max(
                float(args.get("low_complexity_threshold", 0.66)),
                float(args.get("high_complexity_threshold", 0.33)),
            )
            high = min(
                float(args.get("low_complexity_threshold", 0.66)),
                float(args.get("high_complexity_threshold", 0.33)),
            )
            candidate = score >= float(args.get("ssm_threshold", 0.0))
            masks = [
                candidate & (variance_score >= low),
                candidate & (variance_score < low) & (variance_score >= high),
                candidate & (variance_score < high),
            ]
            keep_counts = []
            for mask, keep in zip(masks, (1, 2, 4)):
                count = int(mask.sum(1).min().item())
                if count:
                    keep_counts.append(
                        (
                            score.masked_fill(~mask, -1e9).topk(count, dim=-1).indices,
                            keep,
                        )
                    )
            if not keep_counts:
                return _empty_plan(x)
        else:
            count = min(target, window_count)
            if count <= 0:
                return _empty_plan(x)
            selected = score.topk(count, dim=-1).indices
            selected = selected.gather(
                1, variance_score.gather(1, selected).argsort(-1, descending=True)
            )
            low_count = int(count * 0.25)
            middle_count = int(count * (0.25 if variant == "legacy_taylor" else 0.30))
            boundary = low_count + middle_count
            keep_counts = []
            if low_count:
                keep_counts.append((selected[:, :low_count], 1))
            if middle_count:
                keep_counts.append((selected[:, low_count:boundary], 2))
            if boundary < count:
                keep_counts.append((selected[:, boundary:], 4))

    random_order = _random_scores(
        info, batch, windows.numel(), x.device, 23 + _segment_salt(segment)
    )
    random_order = random_order.view(batch, window_count, size).argsort(-1)
    expanded = windows.unsqueeze(0).expand(batch, -1, -1)
    shuffled = expanded.gather(2, random_order)
    selected_mask = torch.zeros(batch, window_count, dtype=torch.bool, device=x.device)
    sources_list, destinations_list, mapping_list = [], [], []
    destination_offset = 0
    for chosen, keep in keep_counts:
        selected_mask.scatter_(1, chosen, True)
        chosen_tokens = shuffled.gather(
            1, chosen.unsqueeze(-1).expand(batch, chosen.shape[1], size)
        )
        destinations, sources = chosen_tokens[:, :, :keep], chosen_tokens[:, :, keep:]
        if keep == 1:
            local_mapping = torch.zeros_like(sources)
        else:
            features = normalized_windows.gather(
                1,
                chosen[:, :, None, None].expand(
                    batch, chosen.shape[1], size, normalized_windows.shape[-1]
                ),
            )
            reorder = random_order.gather(
                1, chosen.unsqueeze(-1).expand(batch, chosen.shape[1], size)
            )
            features = features.gather(2, reorder.unsqueeze(-1).expand_as(features))
            local_mapping = (
                features[:, :, keep:] @ features[:, :, :keep].transpose(-1, -2)
            ).argmax(-1)
        base = torch.arange(chosen.shape[1], device=x.device).view(1, -1, 1) * keep
        sources_list.append(sources.reshape(batch, -1))
        destinations_list.append(destinations.reshape(batch, -1))
        mapping_list.append(
            (destination_offset + base + local_mapping).reshape(batch, -1)
        )
        destination_offset += destinations.shape[1] * destinations.shape[2]

    unselected_count = int((~selected_mask).sum(1).min().item())
    if unselected_count:
        indices = (~selected_mask).float().topk(unselected_count, dim=-1).indices
        unmerged = expanded.gather(
            1, indices.unsqueeze(-1).expand(batch, -1, size)
        ).reshape(batch, -1)
    else:
        unmerged = torch.empty(batch, 0, dtype=torch.long, device=x.device)
    if border.numel():
        unmerged = torch.cat([unmerged, border.view(1, -1).expand(batch, -1)], dim=1)
    sources = torch.cat(sources_list, 1)
    destinations = torch.cat(destinations_list, 1)
    mapping = torch.cat(mapping_list, 1)
    plan = _plan_from_groups(tokens, unmerged, destinations, sources, mapping)
    _update_independence(info, plan, bool(args.get("pseudo_merge", False)), segment)
    return plan


def make_fidm_plan(
    x: Tensor,
    height: int,
    width: int,
    info: Dict,
    use_slic: bool,
    segment: str = "target",
) -> MergePlan:
    batch, tokens, _ = x.shape
    args = info["args"]
    windows, border = _window_indices(
        height, width, int(args["sy"]), int(args["sx"]), x.device
    )
    if windows.numel() == 0 or windows.shape[1] <= 1:
        return _empty_plan(x)
    window_count, size = windows.shape
    target_remove = int(tokens * float(info["state"].get("ratio_current", 0.0)))
    if target_remove <= 0:
        return _empty_plan(x)
    expanded = windows.unsqueeze(0).expand(batch, -1, -1)
    last = _get_independence(info, segment)
    if last is not None and last.shape == (batch, tokens):
        positions = (
            _gather_tokens(last.unsqueeze(-1), expanded.reshape(batch, -1, 1))
            .view(batch, window_count, size)
            .argmax(-1, keepdim=True)
        )
    elif not bool(args.get("use_rand", True)):
        positions = torch.zeros(
            batch, window_count, 1, dtype=torch.long, device=x.device
        )
    else:
        positions = (
            _random_scores(
                info, batch, windows.numel(), x.device, 37 + _segment_salt(segment)
            )
            .view(batch, window_count, size)
            .argmax(-1, keepdim=True)
        )
    destinations = expanded.gather(2, positions).squeeze(-1)
    mask = torch.ones(batch, window_count, size, dtype=torch.bool, device=x.device)
    mask.scatter_(2, positions, False)
    candidates = expanded[mask].view(batch, -1)
    normalized = F.normalize(x.float(), dim=-1).to(x.dtype)
    scores = _gather_tokens(normalized, candidates.unsqueeze(-1)) @ _gather_tokens(
        normalized, destinations.unsqueeze(-1)
    ).transpose(1, 2)
    if use_slic:
        yy, xx = torch.meshgrid(
            torch.arange(height, device=x.device),
            torch.arange(width, device=x.device),
            indexing="ij",
        )
        coordinates = (
            torch.stack(
                [xx.float() / max(width - 1, 1), yy.float() / max(height - 1, 1)], -1
            )
            .view(1, tokens, 2)
            .expand(batch, -1, -1)
        )
        scores = scores - float(args.get("slic_alpha", 0.5)) * torch.cdist(
            _gather_tokens(coordinates, candidates.unsqueeze(-1)),
            _gather_tokens(coordinates, destinations.unsqueeze(-1)),
        )
    best_score, best_destination = scores.max(-1)
    remove = min(target_remove, candidates.shape[1])
    if tokens >= 256:
        remove = remove // 16 * 16
    if remove <= 0:
        return _empty_plan(x)
    order = best_score.argsort(-1, descending=True)
    sources = candidates.gather(1, order[:, :remove])
    unmerged = candidates.gather(1, order[:, remove:])
    if border.numel():
        unmerged = torch.cat([unmerged, border.view(1, -1).expand(batch, -1)], dim=1)
    mapping = best_destination.gather(1, order[:, :remove])
    plan = _plan_from_groups(tokens, unmerged, destinations, sources, mapping)
    _update_independence(info, plan, bool(args.get("pseudo_merge", False)), segment)
    return plan


def compute_ratio(info: Dict) -> float:
    args, state = info["args"], info["state"]
    ratio, deviation = float(args.get("ratio", 0.3)), float(args.get("deviation", 0.2))
    steps = max(int(state.get("num_steps", 1)), 1)
    progress = int(state.get("step", 0)) / max(steps - 1, 1)
    return max(
        0.0,
        min(
            1.0,
            (ratio - deviation) + 2.0 * deviation * math.cos(progress * math.pi / 2),
        ),
    )


def _is_protected(info: Dict) -> bool:
    args, state = info["args"], info["state"]
    step_frequency, layer_frequency = (
        args.get("protect_steps_frequency"),
        args.get("protect_layers_frequency"),
    )
    step, layer, count = (
        int(state.get("step", 0)),
        int(state.get("global_layer", 0)),
        int(state.get("layer_count", 1)),
    )
    step_protected = (
        step_frequency is not None
        and int(step_frequency) > 0
        and step % int(step_frequency) == 0
    )
    layer_protected = (
        layer_frequency is not None
        and int(layer_frequency) > 0
        and (layer % int(layer_frequency) == 0 or layer == count - 1)
    )
    if layer_protected:
        branch = str(state.get("cfg_branch", "cond"))
        state["last_independent"] = {
            key: value
            for key, value in state.get("last_independent", {}).items()
            if key[0] != branch
        }
    return step_protected or layer_protected


def _merge_flags(info: Dict) -> Tuple[bool, bool, bool]:
    if _is_protected(info):
        return False, False, True
    if info["method"] == "ToMe":
        return (
            bool(info["args"].get("merge_attn", True)),
            bool(info["args"].get("merge_mlp", False)),
            False,
        )
    early = int(info["state"].get("step", 0)) <= int(
        info["args"].get("switch_step", 20)
    )
    return early, True, False


def _normalize_segment_shapes(
    img_shapes: Any, batch: int, tokens: int
) -> List[Tuple[str, int, int, int]]:
    if img_shapes is None:
        raise ValueError("Qwen SDTM requires img_shapes")
    rows = img_shapes
    if isinstance(rows, tuple):
        rows = list(rows)
    if not isinstance(rows, list) or not rows:
        raise ValueError("img_shapes must contain one shape list per batch sample")
    nested = (
        isinstance(rows[0], (list, tuple))
        and rows[0]
        and isinstance(rows[0][0], (list, tuple))
    )
    batch_rows = rows if nested else [rows]
    if len(batch_rows) not in {1, batch}:
        raise ValueError(
            f"img_shapes has {len(batch_rows)} rows for batch size {batch}"
        )
    first = batch_rows[0]
    normalized = [tuple(int(value) for value in shape) for shape in first]
    if len(normalized) > 2:
        raise NotImplementedError(
            "only one target and one reference image are supported"
        )
    if len(batch_rows) == batch:
        for row in batch_rows[1:]:
            if [tuple(int(value) for value in shape) for shape in row] != normalized:
                raise ValueError(
                    "all samples in a Qwen batch must share target/reference token shapes"
                )
    result = []
    for index, (frames, height, width) in enumerate(normalized):
        if frames != 1:
            raise NotImplementedError(
                "video or multi-frame Qwen inputs are outside this migration"
            )
        result.append(
            (
                "target" if index == 0 else "reference",
                frames * height * width,
                height,
                width,
            )
        )
    if sum(item[1] for item in result) != tokens:
        raise ValueError(
            f"img_shapes describe {sum(item[1] for item in result)} tokens, but input has {tokens}"
        )
    return result


def _make_local_plan(
    x: Tensor, height: int, width: int, info: Dict, segment: str
) -> MergePlan:
    if info["method"] == "ToMe":
        return make_tome_plan(x, height, width, info, segment)
    info["state"]["ratio_current"] = compute_ratio(info)
    if int(info["state"].get("step", 0)) <= int(info["args"].get("switch_step", 20)):
        return make_ssm_plan(x, height, width, info, info["variant"], segment)
    return make_fidm_plan(
        x, height, width, info, info["variant"] == "testidm2", segment
    )


def _make_composite_plan(x: Tensor, info: Dict) -> CompositeMergePlan:
    shapes = info["state"]["segment_shapes"]
    scope = info["args"].get("token_scope", "target")
    segments, offset = [], 0
    for name, length, height, width in shapes:
        value = x[:, offset : offset + length]
        plan = (
            _make_local_plan(value, height, width, info, name)
            if name == "target" or scope == "all-separate"
            else None
        )
        segments.append(SegmentPlan(name, length, plan))
        offset += length
    composite = CompositeMergePlan(segments, x.shape[0])
    state = info["state"]
    state["plan_count"] = int(state.get("plan_count", 0)) + 1
    state.setdefault("reduced_token_counts", []).append(composite.reduced_tokens)
    return composite


def _apply_complex_rope(x: Tensor, frequencies: Tensor) -> Tensor:
    complex_x = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    if frequencies.ndim == 2:
        frequencies = frequencies[None, :, None, :]
    elif frequencies.ndim == 3:
        frequencies = frequencies[:, :, None, :]
    else:
        raise ValueError(f"unsupported Qwen RoPE shape: {tuple(frequencies.shape)}")
    return (
        torch.view_as_real(complex_x * frequencies.to(x.device)).flatten(-2).type_as(x)
    )


class BatchedQwenAttnProcessor2_0:
    """Qwen joint attention with batch-aware complex RoPE and explicit masks."""

    _attention_backend = None

    def __init__(self, use_encoder_hidden_states_mask: bool = True):
        # Diffusers 0.35.1's native Qwen processor does not consume this mask.
        # Accelerated modes enable it because merged batches need an explicit
        # text/image key mask; the Default compatibility path leaves it off to
        # preserve the official processor's behavior.
        self.use_encoder_hidden_states_mask = use_encoder_hidden_states_mask

    def __call__(
        self,
        attn,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_hidden_states_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        image_rotary_emb: Optional[Tuple[Tensor, Tensor]] = None,
        **kwargs,
    ):
        if kwargs:
            raise NotImplementedError(
                f"unsupported Qwen attention options: {', '.join(sorted(kwargs))}"
            )
        if encoder_hidden_states is None:
            raise ValueError("Qwen joint attention requires encoder_hidden_states")
        batch, image_length = hidden_states.shape[:2]
        text_length = encoder_hidden_states.shape[1]

        def heads(value: Tensor) -> Tensor:
            return value.unflatten(-1, (attn.heads, -1))

        image_q, image_k, image_v = (
            heads(attn.to_q(hidden_states)),
            heads(attn.to_k(hidden_states)),
            heads(attn.to_v(hidden_states)),
        )
        text_q, text_k, text_v = (
            heads(attn.add_q_proj(encoder_hidden_states)),
            heads(attn.add_k_proj(encoder_hidden_states)),
            heads(attn.add_v_proj(encoder_hidden_states)),
        )
        if attn.norm_q is not None:
            image_q = attn.norm_q(image_q)
        if attn.norm_k is not None:
            image_k = attn.norm_k(image_k)
        if attn.norm_added_q is not None:
            text_q = attn.norm_added_q(text_q)
        if attn.norm_added_k is not None:
            text_k = attn.norm_added_k(text_k)
        if image_rotary_emb is not None:
            image_frequencies, text_frequencies = image_rotary_emb
            image_q, image_k = (
                _apply_complex_rope(image_q, image_frequencies),
                _apply_complex_rope(image_k, image_frequencies),
            )
            text_q, text_k = (
                _apply_complex_rope(text_q, text_frequencies),
                _apply_complex_rope(text_k, text_frequencies),
            )

        query = torch.cat([text_q, image_q], 1).transpose(1, 2)
        key = torch.cat([text_k, image_k], 1).transpose(1, 2)
        value = torch.cat([text_v, image_v], 1).transpose(1, 2)
        if attention_mask is not None:
            mask = attention_mask
        elif (
            self.use_encoder_hidden_states_mask
            and encoder_hidden_states_mask is not None
        ):
            text_mask = encoder_hidden_states_mask.to(
                device=query.device, dtype=torch.bool
            )
            image_mask = torch.ones(
                batch, image_length, device=query.device, dtype=torch.bool
            )
            mask = torch.cat([text_mask, image_mask], 1)[:, None, None, :]
        else:
            mask = None
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False
        )
        output = output.transpose(1, 2).flatten(2, 3).to(query.dtype)
        text_output, image_output = output[:, :text_length], output[:, text_length:]
        image_output = attn.to_out[0](image_output)
        if len(attn.to_out) > 1:
            image_output = attn.to_out[1](image_output)
        return image_output, attn.to_add_out(text_output)


def _merge_rope(
    rotary: Tuple[Tensor, Tensor], plan: CompositeMergePlan
) -> Tuple[Tensor, Tensor]:
    image, text = rotary
    return plan.gather_image_rope(image), text


def _feature_key(info: Dict, branch: str) -> Tuple[str, int, str]:
    state = info["state"]
    return str(state.get("cfg_branch", "cond")), int(state["layer"]), branch


def _cached(info: Dict, branch: str) -> Optional[Tensor]:
    return info["features"].get(_feature_key(info, branch))


def _store(info: Dict, branch: str, value: Tensor, force: bool = False) -> None:
    if force or bool(info["args"].get("cache_each_step", False)):
        info["features"][_feature_key(info, branch)] = value.detach()


def _unmerge(
    info: Dict, branch: str, plan: Optional[CompositeMergePlan], value: Tensor
) -> Tensor:
    return (
        value
        if plan is None
        else plan.unmerge(
            value, _cached(info, branch), float(info["args"].get("mcw", 1.0))
        )
    )


def _block_forward(
    self,
    hidden_states: Tensor,
    encoder_hidden_states: Tensor,
    encoder_hidden_states_mask: Tensor,
    temb: Tensor,
    image_rotary_emb: Optional[Tuple[Tensor, Tensor]] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Tensor, Tensor]:
    info = self._qwen_sdtm_info
    layer = self._qwen_sdtm_layer
    info["state"].update(layer=layer, global_layer=layer)
    info["state"]["executed_blocks"] = int(info["state"].get("executed_blocks", 0)) + 1
    if info["state"].get("taylor_type") == "Taylor":
        for name, target in (
            ("image_attn", "image"),
            ("image_mlp", "image"),
            ("text_attn", "text"),
            ("text_mlp", "text"),
        ):
            value = taylor_formula(info, _feature_key(info, name))
            if value is not None:
                if target == "image":
                    hidden_states = hidden_states + value
                else:
                    encoder_hidden_states = encoder_hidden_states + value
        return encoder_hidden_states, hidden_states

    attention_kwargs = dict(joint_attention_kwargs or {})
    if attention_kwargs:
        raise NotImplementedError(
            "LoRA/IP-Adapter attention kwargs are outside this Qwen migration"
        )
    merge_attention, merge_mlp, protected = _merge_flags(info)
    image_mod1, image_mod2 = self.img_mod(temb).chunk(2, dim=-1)
    text_mod1, text_mod2 = self.txt_mod(temb).chunk(2, dim=-1)
    image_modulated, image_gate1 = self._modulate(
        self.img_norm1(hidden_states), image_mod1
    )
    text_modulated, text_gate1 = self._modulate(
        self.txt_norm1(encoder_hidden_states), text_mod1
    )

    attention_plan = (
        _make_composite_plan(image_modulated, info) if merge_attention else None
    )
    attention_image = (
        attention_plan.apply(
            image_modulated, bool(info["args"].get("pseudo_merge", False))
        )
        if attention_plan
        else image_modulated
    )
    attention_rope = (
        _merge_rope(image_rotary_emb, attention_plan)
        if attention_plan
        else image_rotary_emb
    )
    image_attn, text_attn = self.attn(
        hidden_states=attention_image,
        encoder_hidden_states=text_modulated,
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        image_rotary_emb=attention_rope,
    )
    image_attn = _unmerge(info, "image_attn", attention_plan, image_gate1 * image_attn)
    text_attn = text_gate1 * text_attn
    _store(info, "image_attn", image_attn, protected)
    _store(info, "text_attn", text_attn, protected)
    hidden_states = hidden_states + image_attn
    encoder_hidden_states = encoder_hidden_states + text_attn

    image_modulated2, image_gate2 = self._modulate(
        self.img_norm2(hidden_states), image_mod2
    )
    mlp_plan = _make_composite_plan(image_modulated2, info) if merge_mlp else None
    mlp_input = (
        mlp_plan.apply(image_modulated2, bool(info["args"].get("pseudo_merge", False)))
        if mlp_plan
        else image_modulated2
    )
    image_mlp = _unmerge(
        info, "image_mlp", mlp_plan, image_gate2 * self.img_mlp(mlp_input)
    )
    _store(info, "image_mlp", image_mlp, protected)
    hidden_states = hidden_states + image_mlp

    text_modulated2, text_gate2 = self._modulate(
        self.txt_norm2(encoder_hidden_states), text_mod2
    )
    text_mlp = text_gate2 * self.txt_mlp(text_modulated2)
    _store(info, "text_mlp", text_mlp, protected)
    encoder_hidden_states = encoder_hidden_states + text_mlp
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    if info["method"] == "SDTM_TaylorSeer":
        for name, value in (
            ("image_attn", image_attn),
            ("image_mlp", image_mlp),
            ("text_attn", text_attn),
            ("text_mlp", text_mlp),
        ):
            update_taylor(info, _feature_key(info, name), value)
    return encoder_hidden_states, hidden_states


def _model_forward(self, *args, **kwargs):
    info = self._qwen_sdtm_info
    hidden = kwargs.get("hidden_states", args[0] if args else None)
    state = info["state"]
    call_index = int(state.get("call_index", 0))
    calls_per_step = int(state.get("calls_per_step", 1))
    state["call_index"] = call_index + 1
    state["step"] = call_index // calls_per_step
    state["cfg_branch"] = "uncond" if calls_per_step == 2 and call_index % 2 else "cond"
    state["plan_counter"] = 0
    state["reduced_token_counts"] = []
    state["segment_shapes"] = _normalize_segment_shapes(
        kwargs.get("img_shapes"), hidden.shape[0], hidden.shape[1]
    )
    state["original_image_tokens"] = hidden.shape[1]
    state["taylor_type"] = (
        taylor_cal_type(info) if info["method"] == "SDTM_TaylorSeer" else "full"
    )
    output = self._qwen_sdtm_original_forward(*args, **kwargs)
    reduced = state.get("reduced_token_counts", [])
    state.setdefault("call_records", []).append(
        {
            "call_index": call_index,
            "step": state["step"],
            "cfg_branch": state["cfg_branch"],
            "kind": state["taylor_type"],
            "original_tokens": hidden.shape[1],
            "minimum_reduced_tokens": min(reduced) if reduced else hidden.shape[1],
        }
    )
    return output


def _make_pipeline_class(parent):
    class QwenImageEditPipelineSDTMPatched(parent):
        _qwen_sdtm_parent = parent

        def __call__(self, *args, **kwargs):
            info = self._qwen_sdtm_info
            negative = kwargs.get("negative_prompt")
            has_negative = negative is not None or (
                kwargs.get("negative_prompt_embeds") is not None
                and kwargs.get("negative_prompt_embeds_mask") is not None
            )
            calls_per_step = (
                2
                if float(kwargs.get("true_cfg_scale", 4.0)) > 1 and has_negative
                else 1
            )
            info["state"].update(
                num_steps=int(kwargs.get("num_inference_steps", 50)),
                calls_per_step=calls_per_step,
                call_index=0,
                step=0,
                cfg_branch="cond",
                plan_counter=0,
                plan_count=0,
                last_independent={},
                executed_blocks=0,
                call_records=[],
            )
            info["features"].clear()
            if info["method"] == "SDTM_TaylorSeer":
                reset_taylor(info)
            return super().__call__(*args, **kwargs)

    return QwenImageEditPipelineSDTMPatched


def _save_forward(module, prefix: str) -> None:
    setattr(module, f"{prefix}_had_instance_forward", "forward" in module.__dict__)
    setattr(
        module, f"{prefix}_previous_instance_forward", module.__dict__.get("forward")
    )
    setattr(module, f"{prefix}_original_forward", module.forward)


def _restore_forward(module, prefix: str) -> None:
    if not hasattr(module, f"{prefix}_original_forward"):
        return
    if getattr(module, f"{prefix}_had_instance_forward"):
        module.forward = getattr(module, f"{prefix}_previous_instance_forward")
    elif "forward" in module.__dict__:
        del module.__dict__["forward"]
    for suffix in (
        "original_forward",
        "had_instance_forward",
        "previous_instance_forward",
    ):
        delattr(module, f"{prefix}_{suffix}")


def remove_patch(pipe) -> Any:
    """Restore all forwards, processors and the pipeline class."""
    transformer = getattr(pipe, "transformer", pipe)
    for block in getattr(transformer, "transformer_blocks", []):
        _restore_forward(block, "_qwen_sdtm")
        if hasattr(block, "_qwen_sdtm_original_processor"):
            block.attn.set_processor(block._qwen_sdtm_original_processor)
            del block._qwen_sdtm_original_processor
        for attribute in ("_qwen_sdtm_info", "_qwen_sdtm_layer"):
            if hasattr(block, attribute):
                delattr(block, attribute)
    _restore_forward(transformer, "_qwen_sdtm")
    if hasattr(transformer, "_qwen_sdtm_info"):
        del transformer._qwen_sdtm_info
    if hasattr(pipe.__class__, "_qwen_sdtm_parent"):
        pipe.__class__ = pipe.__class__._qwen_sdtm_parent
    if hasattr(pipe, "_qwen_sdtm_info"):
        del pipe._qwen_sdtm_info
    return pipe


def apply_patch(pipe, method: str, variant: str, **parameters):
    ratio = float(parameters.get("ratio", 0.0))
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("token ratio must be in [0, 1]")
    if method != "ToMe":
        deviation = float(parameters.get("deviation", 0.0))
        if deviation < 0 or ratio - deviation < 0 or ratio + deviation > 1:
            raise ValueError(
                "the complete SDTM ratio +/- deviation schedule must remain in [0, 1]"
            )
    if parameters.get("token_scope", "target") not in {"target", "all-separate"}:
        raise ValueError("token_scope must be 'target' or 'all-separate'")
    if int(parameters.get("sx", 0)) <= 0 or int(parameters.get("sy", 0)) <= 0:
        raise ValueError("merge window dimensions must be positive")
    if method == "SDTM_TaylorSeer" and (
        int(parameters.get("taylor_interval", 0)) <= 0
        or int(parameters.get("taylor_max_order", -1)) < 0
    ):
        raise ValueError("invalid TaylorSeer configuration")
    if not hasattr(pipe, "transformer"):
        raise TypeError("apply_patch expects a QwenImageEditPipeline")
    remove_patch(pipe)
    transformer = pipe.transformer
    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        raise TypeError(
            "pipeline does not contain QwenImageTransformer2DModel.transformer_blocks"
        )
    info = {
        "method": method,
        "variant": variant,
        "args": parameters,
        "state": {
            "num_steps": 50,
            "calls_per_step": 2,
            "call_index": 0,
            "step": 0,
            "cfg_branch": "cond",
            "layer_count": len(blocks),
            "last_independent": {},
            "sample_seeds": None,
        },
        "features": {},
    }
    if method == "SDTM_TaylorSeer":
        info["taylor"] = {
            "interval": int(parameters.get("taylor_interval", 2)),
            "max_order": int(parameters.get("taylor_max_order", 1)),
            "first_enhance": int(parameters.get("taylor_first_enhance", 12)),
            "branches": {},
        }
    pipe._qwen_sdtm_info = transformer._qwen_sdtm_info = info
    pipe.__class__ = _make_pipeline_class(pipe.__class__)
    _save_forward(transformer, "_qwen_sdtm")
    transformer.forward = types.MethodType(_model_forward, transformer)
    for index, block in enumerate(blocks):
        block._qwen_sdtm_info, block._qwen_sdtm_layer = info, index
        _save_forward(block, "_qwen_sdtm")
        block.forward = types.MethodType(_block_forward, block)
        block._qwen_sdtm_original_processor = block.attn.get_processor()
        block.attn.set_processor(BatchedQwenAttnProcessor2_0())
    return pipe


def set_batch_seeds(pipe, seeds: Sequence[int]) -> None:
    info = getattr(pipe, "_qwen_sdtm_info", None)
    if info is not None:
        info["state"]["sample_seeds"] = [int(seed) for seed in seeds]


def patch_statistics(pipe) -> Dict[str, Any]:
    info = getattr(pipe, "_qwen_sdtm_info", None)
    if info is None:
        return {
            "patched": False,
            "sdpa_compatibility_processor": hasattr(
                pipe, "_qwen_sdtm_compat_processors"
            ),
        }
    state = info["state"]
    branches = info.get("taylor", {}).get("branches", {})
    return {
        "patched": True,
        "method": info["method"],
        "variant": info["variant"],
        "token_scope": info["args"].get("token_scope", "target"),
        "layers": state["layer_count"],
        "executed_blocks": state.get("executed_blocks", 0),
        "transformer_calls": state.get("call_index", 0),
        "calls_per_step": state.get("calls_per_step", 1),
        "last_step": state.get("step"),
        "last_cfg_branch": state.get("cfg_branch"),
        "last_taylor_type": state.get("taylor_type"),
        "segment_shapes": state.get("segment_shapes"),
        "plan_count": state.get("plan_count", 0),
        "call_records": state.get("call_records", []),
        "taylor_activated_steps": {
            name: value.get("activated_steps", []) for name, value in branches.items()
        },
    }


def install_counting_processors(transformer) -> List[Tuple[Any, Any]]:
    """Install direct PyTorch SDPA processors so FlopCounterMode sees attention."""
    saved = []
    for block in transformer.transformer_blocks:
        saved.append((block.attn, block.attn.get_processor()))
        block.attn.set_processor(BatchedQwenAttnProcessor2_0())
    return saved


def restore_counting_processors(saved: Sequence[Tuple[Any, Any]]) -> None:
    for attention, processor in saved:
        attention.set_processor(processor)


def install_qwen_sdpa_compatibility(pipe):
    """Replace Diffusers' dispatch wrapper with equivalent direct SDPA.

    Diffusers 0.35.1 always forwards ``enable_gqa=False`` to PyTorch SDPA,
    although PyTorch 2.4 does not yet accept that keyword. Qwen's image
    attention has matching query/key head counts, so omitting the false-valued
    keyword is exactly equivalent. This also keeps the Default mode's original
    behavior of not applying ``encoder_hidden_states_mask``.
    """
    if hasattr(pipe, "_qwen_sdtm_compat_processors"):
        return pipe
    saved = []
    for block in pipe.transformer.transformer_blocks:
        saved.append((block.attn, block.attn.get_processor()))
        block.attn.set_processor(
            BatchedQwenAttnProcessor2_0(use_encoder_hidden_states_mask=False)
        )
    pipe._qwen_sdtm_compat_processors = saved
    return pipe


def remove_qwen_sdpa_compatibility(pipe):
    saved = getattr(pipe, "_qwen_sdtm_compat_processors", None)
    if saved is not None:
        restore_counting_processors(saved)
        del pipe._qwen_sdtm_compat_processors
    return pipe


__all__ = [
    "BatchedQwenAttnProcessor2_0",
    "CompositeMergePlan",
    "MergePlan",
    "apply_patch",
    "install_counting_processors",
    "install_qwen_sdpa_compatibility",
    "patch_statistics",
    "remove_patch",
    "remove_qwen_sdpa_compatibility",
    "restore_counting_processors",
    "set_batch_seeds",
]
