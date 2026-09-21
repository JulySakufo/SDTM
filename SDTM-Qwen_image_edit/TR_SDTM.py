"""Original SDTM strategy adapted to Qwen-Image-Edit's joint transformer."""

from __future__ import annotations

from qwen_core import apply_patch, patch_statistics, remove_patch, set_batch_seeds


def apply_SDTM(
    pipe,
    ratio: float = 0.3,
    deviation: float = 0.2,
    switch_step: int = 20,
    use_rand: bool = True,
    sx: int = 4,
    sy: int = 4,
    a_s: float = 0.05,
    a_d: float = 0.05,
    a_p: float = 2.0,
    pseudo_merge: bool = False,
    mcw: float = 0.1,
    protect_steps_frequency: int = 3,
    protect_layers_frequency: int = -1,
    merge_attn: bool = False,
    merge_mlp: bool = False,
    cache_each_step: bool = True,
    seed: int = 0,
    token_scope: str = "target",
):
    del merge_attn, merge_mlp
    return apply_patch(
        pipe,
        method="SDTM",
        variant="basic",
        ratio=ratio,
        deviation=deviation,
        switch_step=switch_step,
        use_rand=use_rand,
        sx=sx,
        sy=sy,
        a_s=a_s,
        a_d=a_d,
        a_p=a_p,
        pseudo_merge=pseudo_merge,
        mcw=mcw,
        protect_steps_frequency=protect_steps_frequency,
        protect_layers_frequency=protect_layers_frequency,
        cache_each_step=cache_each_step,
        seed=seed,
        token_scope=token_scope,
    )


__all__ = ["apply_SDTM", "remove_patch", "set_batch_seeds", "patch_statistics"]
