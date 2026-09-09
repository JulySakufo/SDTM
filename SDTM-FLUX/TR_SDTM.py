"""Original SDTM strategy adapted to both FLUX transformer streams."""

from __future__ import annotations

from flux_core import apply_patch, remove_patch


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
):
    del merge_attn, merge_mlp  # phase-specific switches are controlled by SDTM
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
    )


__all__ = ["apply_SDTM", "remove_patch"]
