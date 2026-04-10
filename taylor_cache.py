"""
TaylorSeer-style caching for Stable Diffusion 3 Medium (adapted from TaylorSeer-DiT).

Core idea: periodically perform "full" computation steps and cache the intermediate
outputs (attn_output, mlp_output) along with their higher-order finite-difference
derivatives across timesteps. On "Taylor" steps in between, skip the expensive
transformer block computation entirely and reconstruct the outputs via Taylor
expansion from the cached derivatives.

This module is model-agnostic and operates on the `_tore_info` dictionary that is
already threaded through the SDTM pipeline.
"""

import math
from typing import Dict, Optional
import torch


# ---------------------------------------------------------------------------
# Determine whether the current timestep should be a *full* computation step
# or a *Taylor* (cache-approximated) step.
# ---------------------------------------------------------------------------

def taylor_cal_type(tore_info: Dict) -> str:
    """Return 'full' or 'Taylor' for the current denoising step.

    A step is 'full' when:
      1. It is one of the first ``first_enhance`` steps (early steps matter most).
      2. It is one of the last ``first_enhance`` steps.
      3. The cache counter has reached the activation interval.
    Otherwise the step is 'Taylor' and the block outputs will be approximated.
    """
    taylor = tore_info.get("taylor", {})
    states = tore_info.get("states", {})

    interval = taylor.get("interval", 4)
    max_order = taylor.get("max_order", 2)
    first_enhance = taylor.get("first_enhance", 2)

    step_current = states.get("step_current", 0)
    step_count = states.get("step_count", 50)
    cache_counter = taylor.get("cache_counter", 0)

    # last N steps (step index small) or first N steps (step index large)
    last_steps = step_current <= (first_enhance - 1)
    first_steps = step_current >= (step_count - first_enhance)

    if first_steps or last_steps or (cache_counter >= interval - 1):
        taylor["cache_counter"] = 0
        # record this step as an "activated" step for derivative distance calc
        taylor.setdefault("activated_steps", [step_count - 1])
        taylor["activated_steps"].append(step_current)
        return "full"
    else:
        taylor["cache_counter"] = cache_counter + 1
        return "Taylor"


# ---------------------------------------------------------------------------
# Taylor cache initialisation for a (layer, module) slot
# ---------------------------------------------------------------------------

def taylor_cache_init(tore_info: Dict, layer: int, module_name: str):
    """Initialise the cache slot for a given (layer, module_name) on the very
    first activated step so that ``derivative_approximation`` can build up the
    difference table.  Subsequent activated steps will simply overwrite.
    """
    taylor = tore_info.get("taylor", {})
    states = tore_info.get("states", {})
    step_current = states.get("step_current", 0)
    step_count = states.get("step_count", 50)

    cache = taylor.setdefault("cache", {})

    # On the very first activated step, create an empty dict for this slot
    if step_current == (step_count - 1):
        cache[(layer, module_name)] = {}


# ---------------------------------------------------------------------------
# Derivative approximation (finite differences over activated steps)
# ---------------------------------------------------------------------------

def taylor_derivative_approximation(
    tore_info: Dict,
    layer: int,
    module_name: str,
    feature: torch.Tensor,
):
    """Compute finite-difference derivatives and store them as Taylor
    coefficients in ``tore_info['taylor']['cache'][(layer, module_name)]``.

    After calling this, the cache slot will contain::

        {0: f(t),  1: f'(t),  2: f''(t)/2!,  ...}

    which can be consumed by ``taylor_formula``.
    """
    taylor = tore_info["taylor"]
    states = tore_info["states"]
    cache = taylor["cache"]
    max_order = taylor.get("max_order", 2)
    first_enhance = taylor.get("first_enhance", 2)

    activated_steps = taylor.get("activated_steps", [])
    step_current = states.get("step_current", 0)
    step_count = states.get("step_count", 50)

    # Distance (in step indices) between the last two activated steps
    if len(activated_steps) >= 2:
        diff_dist = activated_steps[-1] - activated_steps[-2]
    else:
        diff_dist = 1  # fallback for the very first step

    if diff_dist == 0:
        diff_dist = 1

    key = (layer, module_name)
    old = cache.get(key, {})

    updated = {0: feature.detach()}

    for i in range(max_order):
        if i in old and step_current < (step_count - first_enhance + 1):
            updated[i + 1] = (updated[i] - old[i]) / diff_dist
        else:
            break

    cache[key] = updated


# ---------------------------------------------------------------------------
# Taylor formula: reconstruct the output from cached derivatives
# ---------------------------------------------------------------------------

def taylor_formula(
    tore_info: Dict,
    layer: int,
    module_name: str,
) -> Optional[torch.Tensor]:
    """Approximate the output of the transformer sub-module at the current
    (non-activated) step using the Taylor expansion built from the last
    activated step.

    Returns ``None`` if the cache is empty (should not happen in normal flow).
    """
    taylor = tore_info["taylor"]
    states = tore_info["states"]
    cache = taylor.get("cache", {})

    step_current = states.get("step_current", 0)
    activated_steps = taylor.get("activated_steps", [])

    key = (layer, module_name)
    coeffs = cache.get(key, {})

    if not coeffs:
        return None

    # x = distance from last activated step (can be negative if counting backwards)
    last_act = activated_steps[-1] if activated_steps else step_current
    x = step_current - last_act

    output = torch.zeros_like(coeffs[0])
    for order, coeff in coeffs.items():
        output = output + (1.0 / math.factorial(order)) * coeff * (x ** order)

    return output
