"""Measure the *executed* transformer FLOPs of the SDTM SD3 pipelines.

This project changes the number of image tokens at runtime and, in the
TaylorSeer variant, can skip most of a transformer block. A static profile of
an unmodified DiT therefore does not describe SDTM. This script runs the real
StableDiffusion3Pipeline and places PyTorch's ``FlopCounterMode`` immediately
around each ``pipe.transformer.forward`` call. It consequently measures the
actual token counts and branches used at every denoising step.

Counted region
--------------
Only the SD3 transformer is counted. Prompt encoding and VAE decoding are not
part of the reported values. ``output_type='latent'`` is used so that VAE
decoding is skipped altogether. PyTorch's counter reports two FLOPs for a
multiply-add and counts matrix multiplications, convolutions, and supported
scaled-dot-product-attention kernels. Like fvcore and most DiT papers, it does
not count every elementwise/indexing operation. The SDTM/IDM selection
overhead is therefore only included when it uses an operator supported by the
counter (for example, its matrix multiplications).

The attention backend can change the executed kernel but not its theoretical
FLOPs. Unlike sample.py, this script deliberately does not enable xFormers:
the PyTorch counter cannot see FLOPs inside xFormers' custom CUDA kernels.

Examples (run in the ``sdtm`` conda environment)
-------------------------------------------------
Measure the baseline and the main SDTM implementation with the settings used
by sample.py::

    conda run -n sdtm python measure_flops.py \
        --modes Default,SDTM --height 1024 --width 1024 \
        --num-inference-steps 50 --batch-size 1

Measure the alternative IDM2 implementation and save all per-step values::

    conda run -n sdtm python measure_flops.py \
        --modes Default,SDTM_testidm2 --print-per-step \
        --output-json flops_testidm2.json

Measure SDTM + TaylorSeer::

    conda run -n sdtm python measure_flops.py \
        --modes Default,SDTM_TaylorSeer \
        --Taylor-interval 2 --Taylor-max-order 1 \
        --Taylor-first-enhance 12

``--batch-size`` is the number of generated images/prompts. With classifier-
free guidance (``--guidance-scale > 1``), SD3 internally doubles this batch for
the transformer. The summary reports both the measured batch total and FLOPs
per generated image.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import types
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import torch
    from torch.utils.flop_counter import FlopCounterMode
except ImportError as exc:  # pragma: no cover - depends on the runtime env
    raise ImportError(
        "PyTorch with torch.utils.flop_counter is required. Run this script "
        "inside the project's `sdtm` conda environment."
    ) from exc

try:
    from diffusers import StableDiffusion3Pipeline
except ImportError as exc:  # pragma: no cover - depends on the runtime env
    raise ImportError(
        "diffusers is required. Run this script inside the project's `sdtm` "
        "conda environment."
    ) from exc


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

DEFAULT_MODEL_PATH = os.path.join(
    _HERE,
    "checkpoints",
    "StableDiffusion",
    "stable-diffusion-3-medium-diffusers",
)

ALL_MODES = ("Default", "SDTM", "SDTM_testidm2", "SDTM_TaylorSeer")
_MODE_ALIASES = {
    "default": "Default",
    "baseline": "Default",
    "sdtm": "SDTM",
    "sdtm_testidm2": "SDTM_testidm2",
    "testidm2": "SDTM_testidm2",
    "sdtm_taylorseer": "SDTM_TaylorSeer",
    "taylorseer": "SDTM_TaylorSeer",
    "sdtm_taylor": "SDTM_TaylorSeer",
}


@dataclass
class StepFlops:
    """FLOP record for one call to the SD3 transformer."""

    call_index: int
    step_index: int
    kind: str
    flops: int
    by_operator: Dict[str, int]


def parse_modes(value: str) -> List[str]:
    """Parse a comma-separated mode list while accepting convenient aliases."""
    result: List[str] = []
    for raw_mode in value.split(","):
        raw_mode = raw_mode.strip()
        if not raw_mode:
            continue
        canonical = _MODE_ALIASES.get(raw_mode.lower())
        if canonical is None:
            raise argparse.ArgumentTypeError(
                f"unknown mode {raw_mode!r}; choose from {', '.join(ALL_MODES)}"
            )
        if canonical not in result:
            result.append(canonical)
    if not result:
        raise argparse.ArgumentTypeError("at least one measurement mode is required")
    return result


def _operator_name(op: Any) -> str:
    name = str(op)
    return name if name.startswith("aten.") else f"aten.{name}"


def _shape(value: Any) -> Optional[List[int]]:
    return list(value.shape) if torch.is_tensor(value) else None


class TransformerFlopRecorder:
    """Wrap one transformer's forward method and count every invocation.

    ``FlopCounterMode`` is entered *inside* the wrapper. This is important:
    putting it around the complete pipeline would also count three text
    encoders, scheduler tensor operations, and (unless latent output is used)
    the VAE.
    """

    def __init__(
        self,
        transformer: torch.nn.Module,
        mode: str,
        protect_freq: Optional[int],
    ):
        self.transformer = transformer
        self.mode = mode
        self.protect_freq = protect_freq
        self.steps: List[StepFlops] = []
        self.hidden_shape: Optional[List[int]] = None
        self.encoder_shape: Optional[List[int]] = None
        self._original_forward = transformer.forward
        self._had_instance_forward = "forward" in transformer.__dict__
        self._old_instance_forward = transformer.__dict__.get("forward")
        self._installed = False

    def _step_kind(self, step_index: int) -> str:
        if self.mode == "Default":
            return "full"

        tore_info = getattr(self.transformer, "_tore_info", {})
        states = tore_info.get("states", {})
        if states.get("taylor_type") == "Taylor":
            return "Taylor"

        freq = self.protect_freq
        if freq is not None and freq > 0 and step_index % freq == 0:
            return "full/protected"
        return "full/SDTM" if self.mode == "SDTM_TaylorSeer" else "SDTM"

    def install(self) -> None:
        if self._installed:
            return

        recorder = self
        original_forward = self._original_forward

        def counted_forward(module_self, *args, **kwargs):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]
            encoder_hidden_states = kwargs.get("encoder_hidden_states")

            if recorder.hidden_shape is None:
                recorder.hidden_shape = _shape(hidden_states)
                recorder.encoder_shape = _shape(encoder_hidden_states)

            # No module list is passed: forward hooks inserted by the module
            # hierarchy mode can interfere with diffusers BaseOutput objects.
            with FlopCounterMode(display=False) as counter:
                output = original_forward(*args, **kwargs)

            counts = counter.get_flop_counts().get("Global", {})
            by_operator = {
                _operator_name(op): int(count) for op, count in counts.items()
            }
            call_index = len(recorder.steps)
            tore_info = getattr(module_self, "_tore_info", {})
            state_step = tore_info.get("states", {}).get("step_current")
            step_index = call_index if state_step is None else int(state_step)
            recorder.steps.append(
                StepFlops(
                    call_index=call_index,
                    step_index=step_index,
                    kind=recorder._step_kind(step_index),
                    flops=sum(by_operator.values()),
                    by_operator=by_operator,
                )
            )
            return output

        self.transformer.forward = types.MethodType(counted_forward, self.transformer)
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        if self._had_instance_forward:
            self.transformer.forward = self._old_instance_forward
        else:
            del self.transformer.__dict__["forward"]
        self._installed = False


def _apply_sdtm_mode(pipe: StableDiffusion3Pipeline, mode: str, args) -> None:
    """Apply exactly one of the three SDTM implementations to a fresh pipe."""
    if mode == "Default":
        return

    common = dict(
        ratio=args.sdtm_ratio,
        deviation=args.sdtm_deviation,
        switch_step=args.sdtm_switch_step,
        use_rand=args.sdtm_use_rand,
        sx=args.sdtm_sx,
        sy=args.sdtm_sy,
        a_s=args.sdtm_a_s,
        a_d=args.sdtm_a_d,
        a_p=args.sdtm_a_p,
        pseudo_merge=args.sdtm_pseudo_merge,
        mcw=args.sdtm_mcw,
        protect_steps_frequency=args.sdtm_protect_steps_frequency,
        protect_layers_frequency=args.sdtm_protect_layers_frequency,
    )

    if mode == "SDTM":
        from TR_SDTM import apply_SDTM

        apply_SDTM(pipe, **common)
    elif mode == "SDTM_testidm2":
        from TR_SDTM_testidm2 import apply_SDTM

        apply_SDTM(
            pipe,
            **common,
            adaptive_ssm=args.sdtm_adaptive_ssm,
            ssm_threshold=args.sdtm_ssm_threshold,
            low_complexity_threshold=args.sdtm_low_complexity_threshold,
            high_complexity_threshold=args.sdtm_high_complexity_threshold,
        )
        # TR_SDTM_testidm2 currently stores this setting internally rather than
        # exposing it in apply_SDTM's signature.
        pipe._tore_info["args"]["slic_alpha"] = args.sdtm_slic_alpha
    elif mode == "SDTM_TaylorSeer":
        from TR_SDTM_TaylorSeer import apply_SDTM_TaylorSeer

        apply_SDTM_TaylorSeer(
            pipe,
            **common,
            adaptive_ssm=args.sdtm_adaptive_ssm,
            ssm_threshold=args.sdtm_ssm_threshold,
            low_complexity_threshold=args.sdtm_low_complexity_threshold,
            high_complexity_threshold=args.sdtm_high_complexity_threshold,
            taylor_interval=args.taylor_interval,
            taylor_max_order=args.taylor_max_order,
            taylor_first_enhance=args.taylor_first_enhance,
        )
    else:  # guarded by parse_modes
        raise ValueError(f"unsupported mode: {mode}")

    # sample.py exposes this option but the current apply_* functions hard-code
    # False. Set it explicitly here so the measurement CLI is unambiguous.
    pipe._tore_info["args"]["cache_each_step"] = args.sdtm_cache_each_step


def _load_pipeline(args) -> StableDiffusion3Pipeline:
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.torch_dtype]

    load_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
    if args.variant:
        load_kwargs["variant"] = args.variant
    if args.local_files_only:
        load_kwargs["local_files_only"] = True

    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, **load_kwargs)
    pipe = pipe.to(args.device)
    pipe.transformer.eval()
    pipe.set_progress_bar_config(disable=not args.show_progress)
    return pipe


def _aggregate_operators(steps: Iterable[StepFlops]) -> Dict[str, int]:
    result: Dict[str, int] = defaultdict(int)
    for step in steps:
        for operator, count in step.by_operator.items():
            result[operator] += count
    return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))


def _kind_summary(steps: Sequence[StepFlops]) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for step in steps:
        groups[step.kind].append(step.flops)
    return {
        kind: {
            "steps": len(values),
            "mean_flops": sum(values) / len(values),
            "min_flops": min(values),
            "max_flops": max(values),
        }
        for kind, values in groups.items()
    }


def measure_mode(mode: str, args) -> Dict[str, Any]:
    print(f"\n[info] Loading {args.model_path!r} for mode={mode} ...")
    pipe = _load_pipeline(args)
    recorder: Optional[TransformerFlopRecorder] = None
    try:
        _apply_sdtm_mode(pipe, mode, args)
        # SDTM creates its merge generator lazily from the device's default RNG
        # state. Reset that state for every mode so random merging is comparable.
        torch.manual_seed(args.seed)
        if str(args.device).startswith("cuda"):
            torch.cuda.manual_seed_all(args.seed)
        recorder = TransformerFlopRecorder(
            pipe.transformer, mode, args.sdtm_protect_steps_frequency
        )
        recorder.install()

        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        prompts = [args.prompt] * args.batch_size
        call_kwargs = dict(
            prompt=prompts,
            generator=generator,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            max_sequence_length=args.max_sequence_length,
            output_type="latent",
        )

        print(
            f"[info] Executing {args.num_inference_steps} denoising steps "
            f"({args.height}x{args.width}, prompt batch={args.batch_size}, "
            f"guidance={args.guidance_scale:g}) ..."
        )
        # PyTorch 2.1's TorchDispatch-based FlopCounterMode is bypassed by
        # inference_mode. no_grad preserves dispatch while still avoiding an
        # autograd graph (and matches sample.py's inference context).
        with torch.no_grad():
            output = pipe(**call_kwargs)
        del output

        if not recorder.steps:
            raise RuntimeError("the transformer was not called; no FLOPs were recorded")
        if len(recorder.steps) != args.num_inference_steps:
            print(
                "[warning] Recorded "
                f"{len(recorder.steps)} transformer calls for "
                f"{args.num_inference_steps} requested inference steps. "
                "The executed-call total is reported."
            )

        total_flops = sum(step.flops for step in recorder.steps)
        effective_batch = (
            recorder.hidden_shape[0] if recorder.hidden_shape is not None else None
        )
        result = {
            "mode": mode,
            "prompt_batch_size": args.batch_size,
            "effective_transformer_batch": effective_batch,
            "hidden_states_shape": recorder.hidden_shape,
            "encoder_hidden_states_shape": recorder.encoder_shape,
            "num_transformer_calls": len(recorder.steps),
            "total_flops": total_flops,
            "average_flops_per_step": total_flops / len(recorder.steps),
            "total_flops_per_image": total_flops / args.batch_size,
            "kind_summary": _kind_summary(recorder.steps),
            "by_operator": _aggregate_operators(recorder.steps),
            "steps": [asdict(step) for step in recorder.steps],
        }
        return result
    finally:
        if recorder is not None:
            recorder.uninstall()
        del pipe
        gc.collect()
        if str(args.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _format_tflops(value: float) -> str:
    return f"{value / 1e12:.4f}"


def print_summary(results: Sequence[Dict[str, Any]], args) -> None:
    print("\n" + "=" * 104)
    print("SD3 transformer FLOPs (executed operations; 1 multiply-add = 2 FLOPs)")
    print(
        f"resolution={args.height}x{args.width}, requested_steps={args.num_inference_steps}, "
        f"prompt_batch={args.batch_size}, guidance_scale={args.guidance_scale:g}, "
        f"dtype={args.torch_dtype}"
    )
    print("Counted region: transformer denoising only (text encoders and VAE excluded)")
    print("=" * 104)
    print(
        f"{'Mode':<23}{'TFLOPs/step':>16}{'TFLOPs/batch':>18}"
        f"{'TFLOPs/image':>18}{'vs Default':>15}{'calls / eff.B':>14}"
    )
    print("-" * 104)

    baseline = next((r for r in results if r["mode"] == "Default"), None)
    for result in results:
        ratio = "-"
        if baseline is not None and result["total_flops_per_image"] > 0:
            ratio = (
                f"{baseline['total_flops_per_image'] / result['total_flops_per_image']:.3f}x"
            )
        calls_and_batch = (
            f"{result['num_transformer_calls']} / "
            f"{result['effective_transformer_batch']}"
        )
        print(
            f"{result['mode']:<23}"
            f"{_format_tflops(result['average_flops_per_step']):>16}"
            f"{_format_tflops(result['total_flops']):>18}"
            f"{_format_tflops(result['total_flops_per_image']):>18}"
            f"{ratio:>15}{calls_and_batch:>14}"
        )
    print("=" * 104)

    for result in results:
        print(f"\n[{result['mode']}] step types")
        print(f"  {'type':<19}{'steps':>7}{'mean TFLOPs':>16}{'min':>13}{'max':>13}")
        for kind, values in result["kind_summary"].items():
            print(
                f"  {kind:<19}{int(values['steps']):>7}"
                f"{_format_tflops(values['mean_flops']):>16}"
                f"{_format_tflops(values['min_flops']):>13}"
                f"{_format_tflops(values['max_flops']):>13}"
            )

        if args.operator_breakdown:
            print(f"\n[{result['mode']}] operator totals")
            for operator, count in result["by_operator"].items():
                print(f"  {operator:<48} {_format_tflops(count):>12} TFLOPs")

        if args.print_per_step:
            print(f"\n[{result['mode']}] per-step FLOPs")
            print(f"  {'call':>5}{'step':>7}  {'type':<19}{'TFLOPs':>13}")
            for step in result["steps"]:
                print(
                    f"  {step['call_index']:>5}{step['step_index']:>7}  "
                    f"{step['kind']:<19}{_format_tflops(step['flops']):>13}"
                )

    print(
        "\nNote: elementwise operations, normalization, indexing, sorting, and "
        "some custom kernels are not in PyTorch's FLOP mapping. Do not enable "
        "xFormers for this measurement because its fused attention kernel is "
        "opaque to the counter."
    )


def _validate_args(args) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_inference_steps <= 0:
        raise ValueError("--num-inference-steps must be positive")
    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be positive")
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError("SD3 image height and width must be divisible by 16")
    if args.sdtm_sx <= 0 or args.sdtm_sy <= 0:
        raise ValueError("--SDTM-sx and --SDTM-sy must be positive")
    if not 0 <= args.sdtm_ratio < 1:
        raise ValueError("--SDTM-ratio must be in [0, 1)")
    if args.sdtm_deviation < 0:
        raise ValueError("--SDTM-deviation cannot be negative")
    if not 0 <= args.sdtm_ratio - args.sdtm_deviation:
        raise ValueError("--SDTM-ratio - --SDTM-deviation cannot be negative")
    if args.sdtm_ratio + args.sdtm_deviation >= 1:
        raise ValueError("--SDTM-ratio + --SDTM-deviation must be less than 1")
    if any(mode != "Default" for mode in args.modes):
        if args.height != args.width:
            raise ValueError("the current SDTM implementations require square images")
        token_grid = args.height // 16
        if token_grid % args.sdtm_sx or token_grid % args.sdtm_sy:
            raise ValueError(
                "the SD3 image-token grid (height / 16) must be divisible by "
                "both --SDTM-sx and --SDTM-sy"
            )
    if args.taylor_interval <= 0:
        raise ValueError("--Taylor-interval must be positive")
    if args.taylor_max_order < 0:
        raise ValueError("--Taylor-max-order cannot be negative")
    if args.taylor_first_enhance < 0:
        raise ValueError("--Taylor-first-enhance cannot be negative")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"--device={args.device!r} was requested, but CUDA is unavailable. "
            "Run on a CUDA node or explicitly select --device cpu (very slow for SD3)."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Measure executed transformer FLOPs for SD3 + SDTM variants.",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--variant", default=None, help="Optional diffusers weight variant, e.g. fp16.")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not access the Hugging Face Hub while loading the model.",
    )
    parser.add_argument("--device", default="cuda", help="PyTorch device, e.g. cuda, cuda:1, or cpu.")
    parser.add_argument(
        "--torch-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--modes",
        type=parse_modes,
        default=parse_modes("Default,SDTM"),
        help=f"Comma-separated subset of: {', '.join(ALL_MODES)}.",
    )
    parser.add_argument("--prompt", default="A photo of a cat")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument(
        "--num-inference-steps",
        "--num_inference_steps",
        dest="num_inference_steps",
        type=int,
        default=50,
    )
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--print-per-step", action="store_true")
    parser.add_argument("--operator-breakdown", action="store_true")
    parser.add_argument("--output-json", default=None)

    # Keep aliases matching sample.py so an experiment command can be copied
    # directly between sampling and FLOP measurement.
    parser.add_argument("--SDTM-ratio", "--sdtm-ratio", dest="sdtm_ratio", type=float, default=0.3)
    parser.add_argument(
        "--SDTM-deviation", "--sdtm-deviation", dest="sdtm_deviation", type=float, default=0.2
    )
    parser.add_argument(
        "--SDTM-switch-step", "--sdtm-switch-step", dest="sdtm_switch_step", type=int, default=20
    )
    parser.add_argument(
        "--SDTM-use-rand",
        "--sdtm-use-rand",
        dest="sdtm_use_rand",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--SDTM-sx", "--sdtm-sx", dest="sdtm_sx", type=int, default=4)
    parser.add_argument("--SDTM-sy", "--sdtm-sy", dest="sdtm_sy", type=int, default=4)
    parser.add_argument("--SDTM-a-s", "--sdtm-a-s", dest="sdtm_a_s", type=float, default=0.05)
    parser.add_argument("--SDTM-a-d", "--sdtm-a-d", dest="sdtm_a_d", type=float, default=0.05)
    parser.add_argument("--SDTM-a-p", "--sdtm-a-p", dest="sdtm_a_p", type=float, default=2.0)
    parser.add_argument(
        "--SDTM-adaptive-ssm",
        "--sdtm-adaptive-ssm",
        dest="sdtm_adaptive_ssm",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--SDTM-ssm-threshold", "--sdtm-ssm-threshold", dest="sdtm_ssm_threshold", type=float, default=0.0
    )
    parser.add_argument(
        "--SDTM-low-complexity-threshold",
        "--sdtm-low-complexity-threshold",
        dest="sdtm_low_complexity_threshold",
        type=float,
        default=0.66,
    )
    parser.add_argument(
        "--SDTM-high-complexity-threshold",
        "--sdtm-high-complexity-threshold",
        dest="sdtm_high_complexity_threshold",
        type=float,
        default=0.33,
    )
    parser.add_argument(
        "--SDTM-pseudo-merge",
        "--sdtm-pseudo-merge",
        dest="sdtm_pseudo_merge",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--SDTM-mcw", "--sdtm-mcw", dest="sdtm_mcw", type=float, default=0.1)
    parser.add_argument(
        "--SDTM-protect-steps-frequency",
        "--sdtm-protect-steps-frequency",
        dest="sdtm_protect_steps_frequency",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--SDTM-protect-layers-frequency",
        "--sdtm-protect-layers-frequency",
        dest="sdtm_protect_layers_frequency",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--SDTM-cache-each-step",
        "--SDTM-cache_each_step",
        "--sdtm-cache-each-step",
        dest="sdtm_cache_each_step",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--SDTM-slic-alpha", "--sdtm-slic-alpha", dest="sdtm_slic_alpha", type=float, default=0.5
    )

    parser.add_argument(
        "--Taylor-interval", "--taylor-interval", dest="taylor_interval", type=int, default=2
    )
    parser.add_argument(
        "--Taylor-max-order", "--taylor-max-order", dest="taylor_max_order", type=int, default=1
    )
    parser.add_argument(
        "--Taylor-first-enhance",
        "--taylor-first-enhance",
        dest="taylor_first_enhance",
        type=int,
        default=12,
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    _validate_args(args)

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    results = [measure_mode(mode, args) for mode in args.modes]
    print_summary(results, args)

    if args.output_json:
        output_path = os.path.abspath(args.output_json)
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = {
            "config": {
                key: value for key, value in vars(args).items() if key != "modes"
            },
            "modes": args.modes,
            "results": results,
        }
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"\n[info] Wrote detailed results to {output_path}")


if __name__ == "__main__":
    main()
