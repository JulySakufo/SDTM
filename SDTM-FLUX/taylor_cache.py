"""TaylorSeer finite-difference cache used by the FLUX SDTM patches.

The scheduling and finite-difference rules intentionally match the legacy
``SDTM/TR_SDTM_TaylorSeer.py`` implementation.  Cached values are full-size,
already-gated residuals, so Taylor steps can skip the complete transformer
block computation and add the approximated residual directly.
"""

from __future__ import annotations

import math
from typing import Dict, Hashable, Optional

import torch


def reset_taylor(info: Dict) -> None:
    state = info.setdefault("taylor", {})
    state["cache_counter"] = 0
    state["activated_steps"] = []
    state["cache"] = {}


def taylor_cal_type(info: Dict) -> str:
    """Return ``full`` or ``Taylor`` for the current denoising step."""
    taylor = info["taylor"]
    states = info["state"]
    interval = int(taylor.get("interval", 4))
    first_enhance = int(taylor.get("first_enhance", 2))
    step = int(states.get("step", 0))
    steps = int(states.get("num_steps", 50))
    counter = int(taylor.get("cache_counter", 0))

    first_steps = step < first_enhance
    last_steps = step >= max(steps - first_enhance, 0)
    if first_steps or last_steps or counter >= interval - 1:
        taylor["cache_counter"] = 0
        taylor.setdefault("activated_steps", []).append(step)
        return "full"

    taylor["cache_counter"] = counter + 1
    return "Taylor"


def taylor_cache_init(info: Dict, key: Hashable) -> None:
    """Initialise a cache slot on the first activated step."""
    if int(info["state"].get("step", 0)) == 0:
        info["taylor"].setdefault("cache", {})[key] = {}


def taylor_derivative_approximation(info: Dict, key: Hashable, feature: torch.Tensor) -> None:
    """Store the value and available finite-difference derivatives."""
    taylor = info["taylor"]
    step = int(info["state"].get("step", 0))
    max_order = int(taylor.get("max_order", 2))
    first_enhance = int(taylor.get("first_enhance", 2))
    activated = taylor.setdefault("activated_steps", [])
    distance = activated[-1] - activated[-2] if len(activated) >= 2 else 1
    distance = distance or 1

    cache = taylor.setdefault("cache", {})
    old = cache.get(key, {})
    updated = {0: feature.detach()}
    for order in range(max_order):
        if order not in old or step < first_enhance:
            break
        updated[order + 1] = (updated[order] - old[order]) / distance
    cache[key] = updated


def taylor_formula(info: Dict, key: Hashable) -> Optional[torch.Tensor]:
    """Reconstruct a cached residual at the current step."""
    taylor = info["taylor"]
    coeffs = taylor.get("cache", {}).get(key, {})
    if not coeffs:
        return None

    step = int(info["state"].get("step", 0))
    activated = taylor.get("activated_steps", [])
    last_activated = activated[-1] if activated else step
    distance = step - last_activated
    output = torch.zeros_like(coeffs[0])
    for order, coefficient in coeffs.items():
        output = output + coefficient * (distance**order) / math.factorial(order)
    return output


def update_taylor(info: Dict, key: Hashable, feature: torch.Tensor) -> None:
    taylor_cache_init(info, key)
    taylor_derivative_approximation(info, key, feature)
