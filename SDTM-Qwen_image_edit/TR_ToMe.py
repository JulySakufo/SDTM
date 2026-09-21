"""ToMe adapter for the original single-image Qwen-Image-Edit pipeline."""

from __future__ import annotations

from qwen_core import apply_patch, patch_statistics, remove_patch, set_batch_seeds


def apply_ToMe(
    pipe,
    ratio: float = 0.9,
    sx: int = 2,
    sy: int = 2,
    use_rand: bool = True,
    merge_attn: bool = True,
    merge_crossattn: bool = False,
    merge_mlp: bool = False,
    change_merge_to_prune: bool = True,
    seed: int = 0,
    token_scope: str = "target",
):
    if merge_crossattn:
        raise ValueError(
            "Qwen uses joint attention; text tokens are intentionally never merged"
        )
    return apply_patch(
        pipe,
        method="ToMe",
        variant="tome",
        ratio=ratio,
        sx=sx,
        sy=sy,
        use_rand=use_rand,
        merge_attn=merge_attn,
        merge_mlp=merge_mlp,
        pseudo_merge=change_merge_to_prune,
        seed=seed,
        token_scope=token_scope,
        mcw=1.0,
        cache_each_step=False,
    )


__all__ = ["apply_ToMe", "remove_patch", "set_batch_seeds", "patch_statistics"]
