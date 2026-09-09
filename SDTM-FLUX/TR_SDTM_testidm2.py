"""Adaptive multi-level SSM plus SLIC-FIDM for FLUX.1-dev."""

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
    adaptive_ssm: bool = False,
    ssm_threshold: float = 0.0,
    low_complexity_threshold: float = 0.66,
    high_complexity_threshold: float = 0.33,
    pseudo_merge: bool = False,
    mcw: float = 0.1,
    protect_steps_frequency: int = 3,
    protect_layers_frequency: int = -1,
    merge_attn: bool = False,
    merge_mlp: bool = False,
    cache_each_step: bool = True,
    slic_alpha: float = 0.5,
    seed: int = 0,
):
    del merge_attn, merge_mlp
    return apply_patch(
        pipe,
        method="SDTM_testidm2",
        variant="testidm2",
        ratio=ratio,
        deviation=deviation,
        switch_step=switch_step,
        use_rand=use_rand,
        sx=sx,
        sy=sy,
        a_s=a_s,
        a_d=a_d,
        a_p=a_p,
        adaptive_ssm=adaptive_ssm,
        ssm_threshold=ssm_threshold,
        low_complexity_threshold=low_complexity_threshold,
        high_complexity_threshold=high_complexity_threshold,
        pseudo_merge=pseudo_merge,
        mcw=mcw,
        protect_steps_frequency=protect_steps_frequency,
        protect_layers_frequency=protect_layers_frequency,
        cache_each_step=cache_each_step,
        slic_alpha=slic_alpha,
        seed=seed,
    )


__all__ = ["apply_SDTM", "remove_patch"]
