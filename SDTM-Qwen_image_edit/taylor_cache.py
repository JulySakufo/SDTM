"""Branch-local TaylorSeer cache for Qwen-Image-Edit.

Qwen true CFG evaluates the transformer twice per denoising step.  The legacy
SDTM schedule is therefore maintained independently for the conditional and
unconditional branches, while both branches use the same denoising-step index.
"""

from __future__ import annotations

import math
from typing import Dict, Hashable, Optional

import torch


def _branch(info: Dict) -> Dict:
    root = info["taylor"]
    name = str(info["state"].get("cfg_branch", "cond"))
    return root.setdefault("branches", {}).setdefault(
        name, {"cache_counter": 0, "activated_steps": [], "cache": {}}
    )


def reset_taylor(info: Dict) -> None:
    info.setdefault("taylor", {})["branches"] = {}


def taylor_cal_type(info: Dict) -> str:
    root, branch, states = info["taylor"], _branch(info), info["state"]
    interval = int(root.get("interval", 4))
    first_enhance = int(root.get("first_enhance", 2))
    step = int(states.get("step", 0))
    steps = int(states.get("num_steps", 50))
    counter = int(branch.get("cache_counter", 0))

    if (
        step < first_enhance
        or step >= max(steps - first_enhance, 0)
        or counter >= interval - 1
    ):
        branch["cache_counter"] = 0
        branch.setdefault("activated_steps", []).append(step)
        return "full"
    branch["cache_counter"] = counter + 1
    return "Taylor"


def taylor_cache_init(info: Dict, key: Hashable) -> None:
    if int(info["state"].get("step", 0)) == 0:
        _branch(info).setdefault("cache", {})[key] = {}


def taylor_derivative_approximation(
    info: Dict, key: Hashable, feature: torch.Tensor
) -> None:
    root, branch = info["taylor"], _branch(info)
    step = int(info["state"].get("step", 0))
    max_order = int(root.get("max_order", 2))
    first_enhance = int(root.get("first_enhance", 2))
    activated = branch.setdefault("activated_steps", [])
    distance = activated[-1] - activated[-2] if len(activated) >= 2 else 1
    distance = distance or 1

    cache = branch.setdefault("cache", {})
    old = cache.get(key, {})
    updated = {0: feature.detach()}
    for order in range(max_order):
        if order not in old or step < first_enhance:
            break
        updated[order + 1] = (updated[order] - old[order]) / distance
    cache[key] = updated


def taylor_formula(info: Dict, key: Hashable) -> Optional[torch.Tensor]:
    branch = _branch(info)
    coefficients = branch.get("cache", {}).get(key, {})
    if not coefficients:
        return None
    step = int(info["state"].get("step", 0))
    activated = branch.get("activated_steps", [])
    distance = step - (activated[-1] if activated else step)
    output = torch.zeros_like(coefficients[0])
    for order, coefficient in coefficients.items():
        output = output + coefficient * (distance**order) / math.factorial(order)
    return output


def update_taylor(info: Dict, key: Hashable, feature: torch.Tensor) -> None:
    taylor_cache_init(info, key)
    taylor_derivative_approximation(info, key, feature)
