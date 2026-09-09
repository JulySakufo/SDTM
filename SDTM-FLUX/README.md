# SDTM-FLUX

This directory ports the SD3-Medium SDTM experiments to the local Diffusers
FLUX.1-dev checkpoint. It does not modify Diffusers or the files in the parent
`SDTM` directory.

## Supported modes

- `Default`: native `FluxPipeline`
- `ToMe`: global bipartite token merging
- `SDTM`: structure-then-detail merging
- `SDTM_testidm2`: multi-level SSM and SLIC-aware FIDM
- `SDTM_TaylorSeer`: the legacy TaylorSeer file's 25/25/50 SSM and cosine FIDM

All accelerated modes patch both the 19 double-stream and 38 single-stream
FLUX blocks. Text tokens are retained; only image tokens are merged. The
custom attention processor supports per-sample RoPE after merging.

## Sampling

Run from this directory in the existing `sdtm` environment:

```bash
conda run -n sdtm python sample.py \
  --tore-type Default \
  --model-path ../../models/FLUX.1-dev \
  --caption-path ../datasets/COCO2017/captions_val2017.json \
  --height 1024 --width 1024 \
  --num-inference-steps 50 --guidance-scale 3.5 \
  --batch-size 1 --seed 0 --warmup-runs 1
```

Change `--tore-type` to run another mode. `sample_testidm2.py` is a
compatibility entry point whose default mode is `SDTM_testidm2`. Sampling is
resumable by default and never rewrites the caption JSON. Each COCO image id
has its own latent and merge seed, shared across experiment modes.

FLUX.1-dev is loaded as `bfloat16` by default. Use `--cpu-offload` on a CUDA
machine that cannot hold the complete pipeline in GPU memory. Offload affects
latency and therefore should be identical across modes in timing comparisons.

## FLOPs

The FLOP script measures only calls to `FluxTransformer2DModel`; text encoders
and the VAE are excluded. It intentionally uses PyTorch SDPA so the FLOP
counter can see attention operations.

```bash
conda run -n sdtm python measure_flops.py \
  --modes Default,ToMe,SDTM,SDTM_testidm2,SDTM_TaylorSeer \
  --height 1024 --width 1024 --num-inference-steps 50 \
  --batch-size 1 --output-json flops_flux.json
```

## CLIP Score and FID

Both evaluators accept one or more `NAME=PATH` image sets and require 5,000
valid images by default.

```bash
conda run -n sdtm python clip_evaluation.py \
  --image-set Default=/path/to/default/images \
  --image-set SDTM=/path/to/sdtm/images \
  --output-json clip_scores_flux.json

conda run -n sdtm python fid_evaluation.py \
  --image-set Default=/path/to/default/images \
  --image-set SDTM=/path/to/sdtm/images \
  --output-json fid_scores_flux.json
```

The scripts reuse `../clip_models/ViT-G-14.pt`, `../inceptions`, and
`../datasets/val2017_center_crop_1024` without downloading files.

## Tests

The tests use tiny Diffusers FLUX blocks and do not load the 12B checkpoint:

```bash
PYTHONPATH=. conda run -n sdtm python -m unittest discover -s tests -v
```

They cover rectangular grids, batch-specific RoPE, native no-op parity,
double/single-stream patching, patch restoration, all acceleration modes, and
the legacy Taylor schedule. Full 1024px generation and metric runs require a
CUDA-visible environment.

## Compatibility boundary

The implementation targets Diffusers 0.33's standard `FluxPipeline` and does
not support ControlNet, IP-Adapter, LoRA attention kwargs, or true CFG. The
legacy `a_d` and `a_p` arguments remain in the CLI and manifests for experiment
compatibility but are not assigned new semantics.
