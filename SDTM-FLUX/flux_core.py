"""FLUX.1-dev token merging patches for ToMe, SDTM and TaylorSeer.

This module targets the Diffusers 0.33 FLUX implementation.  It patches only
runtime instances, never the installed Diffusers package.  Image tokens may be
merged independently for every sample in a batch; text tokens are never
removed.  A small attention processor extends FLUX RoPE broadcasting to accept
per-sample positional embeddings after token merging.
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
    """A reversible mapping from a full image-token grid to representatives."""

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
        out = _gather_tokens(x, self.kept_idx)
        if self.source_idx.shape[1] == 0:
            return out
        src = _gather_tokens(x, self.source_idx)
        return out.scatter_reduce(
            1,
            self.source_to_reduced.expand(src.shape[0], src.shape[1], src.shape[-1]),
            src,
            reduce=mode,
            include_self=True,
        )

    def prune(self, x: Tensor) -> Tensor:
        return _gather_tokens(x, self.kept_idx)

    def apply(self, x: Tensor, pseudo_merge: bool = False) -> Tensor:
        return self.prune(x) if pseudo_merge else self.merge(x)

    def unmerge(self, x: Tensor, cache: Optional[Tensor] = None, mcw: float = 1.0) -> Tensor:
        batch, _, channels = x.shape
        kept_values = x
        source_values = _gather_tokens(x, self.source_to_reduced)

        if cache is not None:
            cache = cache.to(device=x.device, dtype=x.dtype)
            if self.destination_idx.shape[1]:
                destination_values = _gather_tokens(x, self.destination_to_reduced)
                cached_destination = _gather_tokens(cache, self.destination_idx)
                destination_values = mcw * destination_values + (1.0 - mcw) * cached_destination
                kept_values = kept_values.clone()
                kept_values.scatter_(
                    1,
                    self.destination_to_reduced.expand(batch, self.destination_to_reduced.shape[1], channels),
                    destination_values,
                )
            if self.source_idx.shape[1]:
                cached_source = _gather_tokens(cache, self.source_idx)
                source_values = mcw * source_values + (1.0 - mcw) * cached_source

        full = torch.zeros(batch, self.num_tokens, channels, device=x.device, dtype=x.dtype)
        full.scatter_(1, self.kept_idx.expand(batch, self.reduced_tokens, channels), kept_values)
        if self.source_idx.shape[1]:
            full.scatter_(1, self.source_idx.expand(batch, self.source_idx.shape[1], channels), source_values)
        return full


def _empty_plan(x: Tensor) -> MergePlan:
    batch, tokens, _ = x.shape
    kept = torch.arange(tokens, device=x.device).view(1, tokens, 1).expand(batch, -1, -1)
    empty = torch.empty(batch, 0, 1, device=x.device, dtype=torch.long)
    return MergePlan(tokens, kept, empty, empty, empty, empty, kept)


def _random_scores(info: Dict, batch: int, length: int, device: torch.device, salt: int) -> Tensor:
    """Create deterministic per-image random values for merge decisions."""
    if not bool(info["args"].get("use_rand", True)):
        return torch.arange(length, device=device, dtype=torch.float32).view(1, -1).expand(batch, -1)
    sample_seeds = info["state"].get("sample_seeds")
    step = int(info["state"].get("step", 0))
    layer = int(info["state"].get("global_layer", 0))
    counter = int(info["state"].get("plan_counter", 0))
    info["state"]["plan_counter"] = counter + 1
    offset = step * 1_000_003 + layer * 10_007 + counter * 101 + salt
    if sample_seeds is None:
        sample_seeds = [int(info["args"].get("seed", 0)) + i for i in range(batch)]

    rows = []
    for index in range(batch):
        generator = torch.Generator(device=device)
        generator.manual_seed((int(sample_seeds[index]) + offset) % (2**63 - 1))
        rows.append(torch.rand(length, device=device, generator=generator))
    return torch.stack(rows, dim=0)


def _window_indices(height: int, width: int, sy: int, sx: int, device: torch.device) -> Tensor:
    if sy <= 0 or sx <= 0 or height % sy or width % sx:
        raise ValueError(
            f"image token grid {height}x{width} must be divisible by window {sy}x{sx}"
        )
    grid = torch.arange(height * width, device=device).view(height, width)
    return (
        grid.view(height // sy, sy, width // sx, sx)
        .permute(0, 2, 1, 3)
        .reshape(-1, sy * sx)
    )


def _plan_from_groups(
    num_tokens: int,
    unmerged: Tensor,
    destinations: Tensor,
    sources: Tensor,
    source_to_destination: Tensor,
) -> MergePlan:
    """Build a plan from original token indices and destination-list slots."""
    batch = unmerged.shape[0]
    kept = torch.cat([unmerged, destinations], dim=1).unsqueeze(-1)
    source_idx = sources.unsqueeze(-1)
    destination_offset = unmerged.shape[1]
    source_to_reduced = (source_to_destination + destination_offset).unsqueeze(-1)
    destination_to_reduced = (
        torch.arange(destinations.shape[1], device=destinations.device)
        .view(1, -1)
        .expand(batch, -1)
        + destination_offset
    ).unsqueeze(-1)
    return MergePlan(
        num_tokens=num_tokens,
        kept_idx=kept,
        source_idx=source_idx,
        source_to_reduced=source_to_reduced,
        destination_idx=destinations.unsqueeze(-1),
        destination_to_reduced=destination_to_reduced,
        unmerged_idx=unmerged.unsqueeze(-1),
    )


def _update_independence(info: Dict, plan: MergePlan, pseudo_merge: bool) -> None:
    batch = plan.batch_size
    previous = info["state"].get("last_independent")
    if previous is None or previous.shape != (batch, plan.num_tokens):
        previous = torch.zeros(batch, plan.num_tokens, device=plan.kept_idx.device, dtype=torch.int32)
    previous.add_(1)
    independent = plan.kept_idx.squeeze(-1) if pseudo_merge else plan.unmerged_idx.squeeze(-1)
    if independent.shape[1]:
        previous.scatter_(1, independent, torch.zeros_like(independent, dtype=previous.dtype))
    info["state"]["last_independent"] = previous


def make_tome_plan(x: Tensor, height: int, width: int, info: Dict) -> MergePlan:
    batch, tokens, _ = x.shape
    args = info["args"]
    retained_ratio = float(args.get("ratio", 0.9))
    if not 0.0 <= retained_ratio <= 1.0:
        raise ValueError("ToMe ratio is the retained-token ratio and must be in [0, 1]")
    remove = int(tokens * (1.0 - retained_ratio))
    if tokens >= 256:
        remove = remove // 16 * 16
    if remove <= 0:
        return _empty_plan(x)

    windows = _window_indices(height, width, int(args["sy"]), int(args["sx"]), x.device)
    window_count, window_size = windows.shape
    random_order = _random_scores(info, batch, window_count * window_size, x.device, 11)
    random_order = random_order.view(batch, window_count, window_size).argsort(dim=-1)
    expanded_windows = windows.unsqueeze(0).expand(batch, -1, -1)
    shuffled = expanded_windows.gather(2, random_order)
    destinations = shuffled[:, :, 0]
    candidates = shuffled[:, :, 1:].reshape(batch, -1)

    normalized = F.normalize(x.float(), dim=-1).to(x.dtype)
    source_features = _gather_tokens(normalized, candidates.unsqueeze(-1))
    destination_features = _gather_tokens(normalized, destinations.unsqueeze(-1))
    scores = source_features @ destination_features.transpose(1, 2)
    best_score, best_destination = scores.max(dim=-1)
    remove = min(remove, candidates.shape[1])
    order = best_score.argsort(dim=-1, descending=True)
    source_order = order[:, :remove]
    unmerged_order = order[:, remove:]
    sources = candidates.gather(1, source_order)
    unmerged = candidates.gather(1, unmerged_order)
    mapping = best_destination.gather(1, source_order)
    return _plan_from_groups(tokens, unmerged, destinations, sources, mapping)


def _window_statistics(x: Tensor, windows: Tensor, info: Dict, improved: bool) -> Tuple[Tensor, Tensor, Tensor]:
    batch, _, channels = x.shape
    window_count, size = windows.shape
    values = _gather_tokens(x, windows.reshape(1, -1, 1).expand(batch, -1, -1))
    values = values.view(batch, window_count, size, channels)
    normalized = F.normalize(values.float(), dim=-1)
    similarities = normalized @ normalized.transpose(-1, -2)
    similarity_score = similarities.mean(dim=(-1, -2))
    variance_raw = similarities.var(dim=(-1, -2), unbiased=False)
    minimum = variance_raw.amin(dim=1, keepdim=True)
    maximum = variance_raw.amax(dim=1, keepdim=True)
    variance_score = 1.0 - (variance_raw - minimum) / (maximum - minimum + 1e-6)

    priority = torch.zeros_like(similarity_score)
    last = info["state"].get("last_independent")
    if last is not None and last.shape[1] == x.shape[1]:
        normalized_priority = last.float() / (last.float().mean(dim=1, keepdim=True) + 1e-6)
        priority = _gather_tokens(
            normalized_priority.unsqueeze(-1), windows.reshape(1, -1, 1).expand(batch, -1, -1)
        ).view(batch, window_count, size).mean(-1)
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
) -> MergePlan:
    batch, tokens, _ = x.shape
    args = info["args"]
    windows = _window_indices(height, width, int(args["sy"]), int(args["sx"]), x.device)
    window_count, size = windows.shape
    if size <= 1:
        return _empty_plan(x)
    improved = variant in {"testidm2", "legacy_taylor"}
    score, variance_score, normalized_windows = _window_statistics(x, windows, info, improved)
    target = max(0, int(tokens * float(info["state"].get("ratio_current", 0.0))))

    if not improved:
        selected_count = min(window_count, target // (size - 1))
        if selected_count <= 0:
            return _empty_plan(x)
        selected = score.topk(selected_count, dim=-1).indices
        keep_counts = [(selected, 1)]
    else:
        if size < 4:
            raise ValueError("SDTM_testidm2 requires a window containing at least four tokens")
        adaptive = bool(args.get("adaptive_ssm", False))
        low_threshold = max(
            float(args.get("low_complexity_threshold", 0.66)),
            float(args.get("high_complexity_threshold", 0.33)),
        )
        high_threshold = min(
            float(args.get("low_complexity_threshold", 0.66)),
            float(args.get("high_complexity_threshold", 0.33)),
        )
        if adaptive:
            candidate = score >= float(args.get("ssm_threshold", 0.0))
            masks = [
                candidate & (variance_score >= low_threshold),
                candidate & (variance_score < low_threshold) & (variance_score >= high_threshold),
                candidate & (variance_score < high_threshold),
            ]
            keep_counts = []
            for mask, keep_n in zip(masks, (1, 2, 4)):
                count = int(mask.sum(1).min().item())
                if count:
                    chosen = score.masked_fill(~mask, -1e9).topk(count, dim=-1).indices
                    keep_counts.append((chosen, keep_n))
        else:
            selected_count = min(target, window_count)
            if selected_count <= 0:
                return _empty_plan(x)
            selected = score.topk(selected_count, dim=-1).indices
            selected_variance = variance_score.gather(1, selected)
            order = selected_variance.argsort(dim=-1, descending=True)
            selected = selected.gather(1, order)
            low_count = int(selected_count * 0.25)
            middle_fraction = 0.25 if variant == "legacy_taylor" else 0.30
            middle_count = int(selected_count * middle_fraction)
            keep_counts = []
            boundaries = (low_count, low_count + middle_count)
            if low_count:
                keep_counts.append((selected[:, : boundaries[0]], 1))
            if middle_count:
                keep_counts.append((selected[:, boundaries[0] : boundaries[1]], 2))
            if boundaries[1] < selected_count:
                keep_counts.append((selected[:, boundaries[1] :], 4))

    random_order = _random_scores(info, batch, window_count * size, x.device, 23)
    random_order = random_order.view(batch, window_count, size).argsort(dim=-1)
    expanded = windows.unsqueeze(0).expand(batch, -1, -1)
    shuffled = expanded.gather(2, random_order)
    selected_mask = torch.zeros(batch, window_count, dtype=torch.bool, device=x.device)
    all_sources: List[Tensor] = []
    all_destinations: List[Tensor] = []
    all_mapping: List[Tensor] = []
    destination_offset = 0

    for chosen_windows, keep_n in keep_counts:
        if chosen_windows.shape[1] == 0:
            continue
        selected_mask.scatter_(1, chosen_windows, True)
        chosen_tokens = shuffled.gather(
            1, chosen_windows.unsqueeze(-1).expand(batch, chosen_windows.shape[1], size)
        )
        destinations = chosen_tokens[:, :, :keep_n]
        sources = chosen_tokens[:, :, keep_n:]
        if keep_n == 1:
            local_mapping = torch.zeros_like(sources)
        else:
            chosen_features = normalized_windows.gather(
                1,
                chosen_windows.unsqueeze(-1).unsqueeze(-1).expand(
                    batch, chosen_windows.shape[1], size, normalized_windows.shape[-1]
                ),
            )
            reorder = random_order.gather(
                1, chosen_windows.unsqueeze(-1).expand(batch, chosen_windows.shape[1], size)
            )
            chosen_features = chosen_features.gather(
                2, reorder.unsqueeze(-1).expand_as(chosen_features)
            )
            local_mapping = (
                chosen_features[:, :, keep_n:] @ chosen_features[:, :, :keep_n].transpose(-1, -2)
            ).argmax(-1)
        window_base = torch.arange(chosen_windows.shape[1], device=x.device).view(1, -1, 1) * keep_n
        mapping = destination_offset + window_base + local_mapping
        all_sources.append(sources.reshape(batch, -1))
        all_destinations.append(destinations.reshape(batch, -1))
        all_mapping.append(mapping.reshape(batch, -1))
        destination_offset += destinations.shape[1] * destinations.shape[2]

    unselected_count = int((~selected_mask).sum(1).min().item())
    if unselected_count:
        unselected_windows = (~selected_mask).float().topk(unselected_count, dim=-1).indices
        unmerged = expanded.gather(
            1, unselected_windows.unsqueeze(-1).expand(batch, unselected_count, size)
        ).reshape(batch, -1)
    else:
        unmerged = torch.empty(batch, 0, dtype=torch.long, device=x.device)
    sources = torch.cat(all_sources, dim=1) if all_sources else torch.empty_like(unmerged[:, :0])
    destinations = torch.cat(all_destinations, dim=1) if all_destinations else torch.empty_like(unmerged[:, :0])
    mapping = torch.cat(all_mapping, dim=1) if all_mapping else torch.empty_like(unmerged[:, :0])
    plan = _plan_from_groups(tokens, unmerged, destinations, sources, mapping)
    _update_independence(info, plan, bool(args.get("pseudo_merge", False)))
    return plan


def make_fidm_plan(
    x: Tensor,
    height: int,
    width: int,
    info: Dict,
    use_slic: bool,
) -> MergePlan:
    batch, tokens, _ = x.shape
    args = info["args"]
    windows = _window_indices(height, width, int(args["sy"]), int(args["sx"]), x.device)
    window_count, size = windows.shape
    expanded = windows.unsqueeze(0).expand(batch, -1, -1)
    last = info["state"].get("last_independent")
    if last is not None and last.shape == (batch, tokens):
        priorities = _gather_tokens(last.unsqueeze(-1), expanded.reshape(batch, -1, 1))
        positions = priorities.view(batch, window_count, size).argmax(-1, keepdim=True)
    elif not bool(args.get("use_rand", True)):
        positions = torch.zeros(batch, window_count, 1, dtype=torch.long, device=x.device)
    else:
        positions = _random_scores(info, batch, window_count * size, x.device, 37)
        positions = positions.view(batch, window_count, size).argmax(-1, keepdim=True)
    destinations = expanded.gather(2, positions).squeeze(-1)
    mask = torch.ones(batch, window_count, size, dtype=torch.bool, device=x.device)
    mask.scatter_(2, positions, False)
    candidates = expanded[mask].view(batch, -1)

    normalized = F.normalize(x.float(), dim=-1).to(x.dtype)
    source_features = _gather_tokens(normalized, candidates.unsqueeze(-1))
    destination_features = _gather_tokens(normalized, destinations.unsqueeze(-1))
    scores = source_features @ destination_features.transpose(1, 2)
    if use_slic:
        yy, xx = torch.meshgrid(
            torch.arange(height, device=x.device), torch.arange(width, device=x.device), indexing="ij"
        )
        coordinates = torch.stack(
            [xx.float() / max(width - 1, 1), yy.float() / max(height - 1, 1)], dim=-1
        ).view(1, tokens, 2).expand(batch, -1, -1)
        source_coordinates = _gather_tokens(coordinates, candidates.unsqueeze(-1))
        destination_coordinates = _gather_tokens(coordinates, destinations.unsqueeze(-1))
        scores = scores - float(args.get("slic_alpha", 0.5)) * torch.cdist(
            source_coordinates, destination_coordinates
        )

    best_score, best_destination = scores.max(dim=-1)
    remove = min(int(tokens * float(info["state"].get("ratio_current", 0.0))), candidates.shape[1])
    if tokens >= 256:
        remove = remove // 16 * 16
    if remove <= 0:
        return _empty_plan(x)
    order = best_score.argsort(dim=-1, descending=True)
    selected_order, unmerged_order = order[:, :remove], order[:, remove:]
    sources = candidates.gather(1, selected_order)
    unmerged = candidates.gather(1, unmerged_order)
    mapping = best_destination.gather(1, selected_order)
    plan = _plan_from_groups(tokens, unmerged, destinations, sources, mapping)
    _update_independence(info, plan, bool(args.get("pseudo_merge", False)))
    return plan


def compute_ratio(info: Dict) -> float:
    args, state = info["args"], info["state"]
    ratio = float(args.get("ratio", 0.3))
    deviation = float(args.get("deviation", 0.2))
    steps = max(int(state.get("num_steps", 1)), 1)
    progress = int(state.get("step", 0)) / max(steps - 1, 1)
    value = (ratio - deviation) + 2.0 * deviation * math.cos(progress * math.pi / 2)
    return max(0.0, min(1.0, value))


def _is_protected(info: Dict) -> bool:
    args, state = info["args"], info["state"]
    step_frequency = args.get("protect_steps_frequency")
    layer_frequency = args.get("protect_layers_frequency")
    step = int(state.get("step", 0))
    layer = int(state.get("global_layer", 0))
    layer_count = int(state.get("layer_count", 1))
    step_protected = step_frequency is not None and int(step_frequency) > 0 and step % int(step_frequency) == 0
    layer_protected = layer_frequency is not None and int(layer_frequency) > 0 and (
        layer % int(layer_frequency) == 0 or layer == layer_count - 1
    )
    if layer_protected:
        state["last_independent"] = None
    return step_protected or layer_protected


def _merge_flags(info: Dict) -> Tuple[bool, bool, bool]:
    if _is_protected(info):
        return False, False, True
    if info["method"] == "ToMe":
        return bool(info["args"].get("merge_attn", True)), bool(info["args"].get("merge_mlp", False)), False
    early = int(info["state"].get("step", 0)) <= int(info["args"].get("switch_step", 20))
    return early, True, False


def _make_plan(x: Tensor, info: Dict) -> MergePlan:
    height, width = info["state"]["grid_size"]
    if x.shape[1] != height * width:
        raise ValueError(f"expected {height * width} image tokens, received {x.shape[1]}")
    if info["method"] == "ToMe":
        return make_tome_plan(x, height, width, info)
    info["state"]["ratio_current"] = compute_ratio(info)
    variant = info["variant"]
    if int(info["state"].get("step", 0)) <= int(info["args"].get("switch_step", 20)):
        return make_ssm_plan(x, height, width, info, variant)
    return make_fidm_plan(x, height, width, info, use_slic=variant == "testidm2")


def _batched_rotary(x: Tensor, rotary: Tuple[Tensor, Tensor]) -> Tensor:
    cos, sin = rotary
    if cos.ndim == 2:
        cos, sin = cos[None, None], sin[None, None]
    elif cos.ndim == 3:
        cos, sin = cos[:, None], sin[:, None]
    else:
        raise ValueError(f"unsupported RoPE shape: {tuple(cos.shape)}")
    cos, sin = cos.to(x.device), sin.to(x.device)
    real, imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    rotated = torch.stack([-imag, real], dim=-1).flatten(3)
    return (x.float() * cos + rotated.float() * sin).to(x.dtype)


class BatchedFluxAttnProcessor2_0:
    """Diffusers FLUX attention with batch-aware RoPE."""

    def __call__(
        self,
        attn,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        image_rotary_emb: Optional[Tuple[Tensor, Tensor]] = None,
        **kwargs,
    ):
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise NotImplementedError(f"unsupported FLUX attention options: {unsupported}")
        batch = hidden_states.shape[0]
        query, key, value = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        def heads(tensor: Tensor) -> Tensor:
            return tensor.view(batch, -1, attn.heads, head_dim).transpose(1, 2)

        query, key, value = heads(query), heads(key), heads(value)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            context_query = heads(attn.add_q_proj(encoder_hidden_states))
            context_key = heads(attn.add_k_proj(encoder_hidden_states))
            context_value = heads(attn.add_v_proj(encoder_hidden_states))
            if attn.norm_added_q is not None:
                context_query = attn.norm_added_q(context_query)
            if attn.norm_added_k is not None:
                context_key = attn.norm_added_k(context_key)
            query = torch.cat([context_query, query], dim=2)
            key = torch.cat([context_key, key], dim=2)
            value = torch.cat([context_value, value], dim=2)

        if image_rotary_emb is not None:
            query = _batched_rotary(query, image_rotary_emb)
            key = _batched_rotary(key, image_rotary_emb)
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        output = output.transpose(1, 2).reshape(batch, -1, attn.heads * head_dim).to(query.dtype)
        if encoder_hidden_states is None:
            return output
        context_length = encoder_hidden_states.shape[1]
        context, image = output[:, :context_length], output[:, context_length:]
        image = attn.to_out[1](attn.to_out[0](image))
        context = attn.to_add_out(context)
        return image, context


def _merge_rope(rotary: Tuple[Tensor, Tensor], plan: MergePlan, text_length: int) -> Tuple[Tensor, Tensor]:
    result = []
    for component in rotary:
        if component.ndim == 2:
            component = component.unsqueeze(0).expand(plan.batch_size, -1, -1)
        text = component[:, :text_length]
        image = component[:, text_length:]
        image = _gather_tokens(image, plan.kept_idx)
        result.append(torch.cat([text, image], dim=1))
    return result[0], result[1]


def _feature_key(info: Dict, branch: str) -> Tuple[str, int, str]:
    return info["state"]["stream"], int(info["state"]["layer"]), branch


def _cached(info: Dict, branch: str) -> Optional[Tensor]:
    return info["features"].get(_feature_key(info, branch))


def _store(info: Dict, branch: str, value: Tensor, force: bool = False) -> None:
    if force or bool(info["args"].get("cache_each_step", False)):
        info["features"][_feature_key(info, branch)] = value.detach()


def _unmerge_branch(info: Dict, branch: str, plan: Optional[MergePlan], value: Tensor) -> Tensor:
    if plan is None:
        return value
    return plan.unmerge(value, _cached(info, branch), float(info["args"].get("mcw", 1.0)))


def _taylor_key(info: Dict, branch: str) -> Tuple[str, int, str]:
    return _feature_key(info, branch)


def _double_forward(
    self,
    hidden_states: Tensor,
    encoder_hidden_states: Tensor,
    temb: Tensor,
    image_rotary_emb=None,
    joint_attention_kwargs=None,
):
    info = self._flux_sdtm_info
    info["state"].update(stream="double", layer=self._flux_sdtm_layer, global_layer=self._flux_sdtm_layer)
    info["state"]["executed_double_blocks"] = int(info["state"].get("executed_double_blocks", 0)) + 1
    if info["state"].get("taylor_type") == "Taylor":
        for branch, target in (("image_attn", "image"), ("image_mlp", "image"), ("text_attn", "text"), ("text_mlp", "text")):
            approximation = taylor_formula(info, _taylor_key(info, branch))
            if approximation is not None:
                if target == "image":
                    hidden_states = hidden_states + approximation
                else:
                    encoder_hidden_states = encoder_hidden_states + approximation
        return encoder_hidden_states, hidden_states

    merge_attn, merge_mlp, protected = _merge_flags(info)
    joint_attention_kwargs = dict(joint_attention_kwargs or {})
    if "ip_hidden_states" in joint_attention_kwargs:
        raise NotImplementedError("IP-Adapter is outside the SDTM-FLUX migration scope")
    norm_image, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
    norm_text, text_gate_attn, text_shift_mlp, text_scale_mlp, text_gate_mlp = self.norm1_context(
        encoder_hidden_states, emb=temb
    )
    text_length = encoder_hidden_states.shape[1]

    attention_plan = _make_plan(norm_image, info) if merge_attn else None
    attention_image = attention_plan.apply(norm_image, bool(info["args"].get("pseudo_merge", False))) if attention_plan else norm_image
    attention_rope = _merge_rope(image_rotary_emb, attention_plan, text_length) if attention_plan else image_rotary_emb
    attention_outputs = self.attn(
        hidden_states=attention_image,
        encoder_hidden_states=norm_text,
        image_rotary_emb=attention_rope,
        **joint_attention_kwargs,
    )
    if len(attention_outputs) != 2:
        raise NotImplementedError("IP-Adapter attention output is not supported")
    image_attn, text_attn = attention_outputs
    image_attn = gate_attn.unsqueeze(1) * image_attn
    image_attn = _unmerge_branch(info, "image_attn", attention_plan, image_attn)
    text_attn = text_gate_attn.unsqueeze(1) * text_attn
    _store(info, "image_attn", image_attn, protected)
    hidden_states = hidden_states + image_attn

    norm_image = self.norm2(hidden_states)
    norm_image = norm_image * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    mlp_plan = _make_plan(norm_image, info) if merge_mlp else None
    mlp_input = mlp_plan.apply(norm_image, bool(info["args"].get("pseudo_merge", False))) if mlp_plan else norm_image
    image_mlp = gate_mlp.unsqueeze(1) * self.ff(mlp_input)
    image_mlp = _unmerge_branch(info, "image_mlp", mlp_plan, image_mlp)
    _store(info, "image_mlp", image_mlp, protected)
    hidden_states = hidden_states + image_mlp

    encoder_hidden_states = encoder_hidden_states + text_attn
    norm_text = self.norm2_context(encoder_hidden_states)
    norm_text = norm_text * (1 + text_scale_mlp[:, None]) + text_shift_mlp[:, None]
    text_mlp = text_gate_mlp.unsqueeze(1) * self.ff_context(norm_text)
    encoder_hidden_states = encoder_hidden_states + text_mlp
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    if info["method"] == "SDTM_TaylorSeer":
        for branch, value in (("image_attn", image_attn), ("image_mlp", image_mlp), ("text_attn", text_attn), ("text_mlp", text_mlp)):
            update_taylor(info, _taylor_key(info, branch), value)
    return encoder_hidden_states, hidden_states


def _single_forward(self, hidden_states: Tensor, temb: Tensor, image_rotary_emb=None, joint_attention_kwargs=None):
    info = self._flux_sdtm_info
    index = self._flux_sdtm_layer
    global_index = int(info["state"]["num_double_layers"]) + index
    info["state"].update(stream="single", layer=index, global_layer=global_index)
    info["state"]["executed_single_blocks"] = int(info["state"].get("executed_single_blocks", 0)) + 1
    residual = hidden_states
    if info["state"].get("taylor_type") == "Taylor":
        approximation = taylor_formula(info, _taylor_key(info, "residual"))
        return hidden_states if approximation is None else hidden_states + approximation

    merge_attn, merge_mlp, protected = _merge_flags(info)
    norm, gate = self.norm(hidden_states, emb=temb)
    text_length = int(info["state"]["text_length"])
    text_norm, image_norm = norm[:, :text_length], norm[:, text_length:]
    joint_attention_kwargs = dict(joint_attention_kwargs or {})
    if joint_attention_kwargs:
        raise NotImplementedError("extra single-stream attention kwargs are outside the migration scope")

    attention_plan = _make_plan(image_norm, info) if merge_attn else None
    attention_image = attention_plan.apply(image_norm, bool(info["args"].get("pseudo_merge", False))) if attention_plan else image_norm
    attention_input = torch.cat([text_norm, attention_image], dim=1)
    attention_rope = _merge_rope(image_rotary_emb, attention_plan, text_length) if attention_plan else image_rotary_emb
    attention = self.attn(hidden_states=attention_input, image_rotary_emb=attention_rope)
    text_attention, image_attention = attention[:, :text_length], attention[:, text_length:]

    mlp_plan = _make_plan(image_norm, info) if merge_mlp else None
    mlp_image_input = mlp_plan.apply(image_norm, bool(info["args"].get("pseudo_merge", False))) if mlp_plan else image_norm
    text_mlp = self.act_mlp(self.proj_mlp(text_norm))
    image_mlp = self.act_mlp(self.proj_mlp(mlp_image_input))

    dimension = text_attention.shape[-1]
    attention_weight = self.proj_out.weight[:, :dimension]
    mlp_weight = self.proj_out.weight[:, dimension:]
    text_projected = F.linear(text_attention, attention_weight) + F.linear(text_mlp, mlp_weight)
    image_attention = F.linear(image_attention, attention_weight)
    image_attention = _unmerge_branch(info, "attention_projected", attention_plan, image_attention)
    image_mlp = F.linear(image_mlp, mlp_weight)
    image_mlp = _unmerge_branch(info, "mlp_projected", mlp_plan, image_mlp)
    _store(info, "attention_projected", image_attention, protected)
    _store(info, "mlp_projected", image_mlp, protected)
    image_projected = image_attention + image_mlp
    projected = torch.cat([text_projected, image_projected], dim=1)
    if self.proj_out.bias is not None:
        projected = projected + self.proj_out.bias
    block_residual = gate.unsqueeze(1) * projected
    hidden_states = residual + block_residual
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    if info["method"] == "SDTM_TaylorSeer":
        update_taylor(info, _taylor_key(info, "residual"), block_residual)
    return hidden_states


def _infer_grid(img_ids: Tensor, image_tokens: int) -> Tuple[int, int]:
    if img_ids is not None and img_ids.ndim == 3:
        img_ids = img_ids[0]
    if img_ids is None or img_ids.shape[0] != image_tokens:
        side = int(math.sqrt(image_tokens))
        if side * side != image_tokens:
            raise ValueError("img_ids are required for a non-square FLUX image-token grid")
        return side, side
    height = int(img_ids[:, 1].max().item()) + 1
    width = int(img_ids[:, 2].max().item()) + 1
    if height * width != image_tokens:
        raise ValueError(f"img_ids describe {height}x{width}, but there are {image_tokens} image tokens")
    return height, width


def _model_forward(self, *args, **kwargs):
    info = self._flux_sdtm_info
    if kwargs.get("controlnet_block_samples") is not None or kwargs.get("controlnet_single_block_samples") is not None:
        raise NotImplementedError("ControlNet is outside the SDTM-FLUX migration scope")
    hidden = kwargs.get("hidden_states", args[0] if args else None)
    encoder = kwargs.get("encoder_hidden_states")
    img_ids = kwargs.get("img_ids")
    state = info["state"]
    state["step"] = int(state.get("call_index", 0))
    state["call_index"] = state["step"] + 1
    state["plan_counter"] = 0
    state["text_length"] = encoder.shape[1]
    state["grid_size"] = _infer_grid(img_ids, hidden.shape[1])
    state["last_independent"] = None if state["step"] == 0 else state.get("last_independent")
    if info["method"] == "SDTM_TaylorSeer":
        state["taylor_type"] = taylor_cal_type(info)
    else:
        state["taylor_type"] = "full"
    return self._flux_sdtm_original_forward(*args, **kwargs)


def _make_pipeline_class(parent):
    class FluxPipelineSDTMPatched(parent):
        _flux_sdtm_parent = parent

        def __call__(self, *args, **kwargs):
            if kwargs.get("true_cfg_scale", 1.0) > 1.0 or kwargs.get("negative_prompt") is not None:
                raise NotImplementedError("true CFG is outside the SDTM-FLUX migration scope")
            info = self._flux_sdtm_info
            steps = kwargs.get("num_inference_steps", 28)
            info["state"].update(
                num_steps=int(steps),
                call_index=0,
                step=0,
                plan_counter=0,
                last_independent=None,
                executed_double_blocks=0,
                executed_single_blocks=0,
            )
            info["features"].clear()
            if info["method"] == "SDTM_TaylorSeer":
                reset_taylor(info)
            return super().__call__(*args, **kwargs)

    return FluxPipelineSDTMPatched


def remove_patch(pipe) -> Any:
    """Restore every forward method, processor and pipeline class we changed."""
    transformer = getattr(pipe, "transformer", pipe)
    for module in transformer.modules():
        if hasattr(module, "_flux_sdtm_original_forward"):
            if getattr(module, "_flux_sdtm_had_instance_forward", False):
                module.forward = module._flux_sdtm_previous_instance_forward
            elif "forward" in module.__dict__:
                del module.__dict__["forward"]
            del module._flux_sdtm_original_forward
        if hasattr(module, "_flux_sdtm_original_processor"):
            module.attn.set_processor(module._flux_sdtm_original_processor)
            del module._flux_sdtm_original_processor
        for attribute in (
            "_flux_sdtm_info",
            "_flux_sdtm_layer",
            "_flux_sdtm_had_instance_forward",
            "_flux_sdtm_previous_instance_forward",
        ):
            if hasattr(module, attribute):
                delattr(module, attribute)
    if hasattr(pipe.__class__, "_flux_sdtm_parent"):
        pipe.__class__ = pipe.__class__._flux_sdtm_parent
    if hasattr(pipe, "_flux_sdtm_info"):
        del pipe._flux_sdtm_info
    return pipe


def apply_patch(pipe, method: str, variant: str, **parameters):
    """Patch a Diffusers ``FluxPipeline`` in place and return it."""
    ratio = float(parameters.get("ratio", 0.0))
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("token ratio must be in [0, 1]")
    if method != "ToMe":
        deviation = float(parameters.get("deviation", 0.0))
        if deviation < 0 or ratio - deviation < 0 or ratio + deviation > 1:
            raise ValueError("the complete SDTM ratio ± deviation schedule must remain in [0, 1]")
    if int(parameters.get("sx", 0)) <= 0 or int(parameters.get("sy", 0)) <= 0:
        raise ValueError("merge window dimensions must be positive")
    if method == "SDTM_TaylorSeer":
        if int(parameters.get("taylor_interval", 0)) <= 0 or int(parameters.get("taylor_max_order", -1)) < 0:
            raise ValueError("invalid TaylorSeer configuration")
    if pipe.__class__.__name__ != "FluxPipeline" and not hasattr(pipe, "transformer"):
        raise TypeError("apply_patch expects a Diffusers FluxPipeline")
    remove_patch(pipe)
    transformer = pipe.transformer
    if not hasattr(transformer, "transformer_blocks") or not hasattr(transformer, "single_transformer_blocks"):
        raise TypeError("the pipeline does not contain a standard FluxTransformer2DModel")

    info = {
        "method": method,
        "variant": variant,
        "args": parameters,
        "state": {
            "num_steps": 50,
            "step": 0,
            "call_index": 0,
            "layer_count": len(transformer.transformer_blocks) + len(transformer.single_transformer_blocks),
            "num_double_layers": len(transformer.transformer_blocks),
            "last_independent": None,
            "sample_seeds": None,
        },
        "features": {},
    }
    if method == "SDTM_TaylorSeer":
        info["taylor"] = {
            "interval": int(parameters.get("taylor_interval", 4)),
            "max_order": int(parameters.get("taylor_max_order", 2)),
            "first_enhance": int(parameters.get("taylor_first_enhance", 2)),
            "cache_counter": 0,
            "activated_steps": [],
            "cache": {},
        }
    pipe._flux_sdtm_info = transformer._flux_sdtm_info = info
    pipe.__class__ = _make_pipeline_class(pipe.__class__)

    transformer._flux_sdtm_had_instance_forward = "forward" in transformer.__dict__
    transformer._flux_sdtm_previous_instance_forward = transformer.__dict__.get("forward")
    transformer._flux_sdtm_original_forward = transformer.forward
    transformer.forward = types.MethodType(_model_forward, transformer)
    for index, block in enumerate(transformer.transformer_blocks):
        block._flux_sdtm_info = info
        block._flux_sdtm_layer = index
        block._flux_sdtm_had_instance_forward = "forward" in block.__dict__
        block._flux_sdtm_previous_instance_forward = block.__dict__.get("forward")
        block._flux_sdtm_original_forward = block.forward
        block.forward = types.MethodType(_double_forward, block)
        block._flux_sdtm_original_processor = block.attn.get_processor()
        block.attn.set_processor(BatchedFluxAttnProcessor2_0())
    for index, block in enumerate(transformer.single_transformer_blocks):
        block._flux_sdtm_info = info
        block._flux_sdtm_layer = index
        block._flux_sdtm_had_instance_forward = "forward" in block.__dict__
        block._flux_sdtm_previous_instance_forward = block.__dict__.get("forward")
        block._flux_sdtm_original_forward = block.forward
        block.forward = types.MethodType(_single_forward, block)
        block._flux_sdtm_original_processor = block.attn.get_processor()
        block.attn.set_processor(BatchedFluxAttnProcessor2_0())
    return pipe


def set_batch_seeds(pipe, seeds: Sequence[int]) -> None:
    info = getattr(pipe, "_flux_sdtm_info", None)
    if info is not None:
        info["state"]["sample_seeds"] = [int(seed) for seed in seeds]


def patch_statistics(pipe) -> Dict[str, Any]:
    info = getattr(pipe, "_flux_sdtm_info", None)
    if info is None:
        return {"patched": False}
    return {
        "patched": True,
        "method": info["method"],
        "variant": info["variant"],
        "double_layers": info["state"]["num_double_layers"],
        "single_layers": info["state"]["layer_count"] - info["state"]["num_double_layers"],
        "executed_double_blocks": info["state"].get("executed_double_blocks", 0),
        "executed_single_blocks": info["state"].get("executed_single_blocks", 0),
        "step": info["state"].get("step"),
        "taylor_type": info["state"].get("taylor_type"),
    }
