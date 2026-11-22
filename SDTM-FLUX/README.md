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

Run from this directory in the existing `sdtm` environment. Use `--tore-type`
for one method, or `--modes all` for all five. `--modes Default,ToMe` runs a
selected comparison. Each method writes its images and `manifest.json` into a
separate experiment directory under `--output-path`.

### Dataset input

```bash
conda run -n sdtm python sample.py \
  --modes all \
  --model-path ../../models/FLUX.1-dev \
  --dataset ../datasets/COCO2017/captions_val2017.json \
  --real-dir ../datasets/val2017_center_crop_1024 \
  --height 1024 --width 1024 \
  --num-inference-steps 50 --guidance-scale 3.5 \
  --batch-size 1 --seed 0 --warmup-runs 1
```

`--caption-path` is an alias for `--dataset`. The JSON can be a COCO object
with `annotations` or a list of `image_id` and `caption` records. The sampler
selects the longest caption per ID. Reference filenames must match the image
IDs, with optional zero padding for numeric IDs. With `--limit N`, both
metrics use only the selected IDs. FID requires at least two selected images.
Dataset input defaults to the local COCO 2017 validation set and its
1024-pixel center crops when no input flag is supplied.

### Custom prompts

```bash
conda run -n sdtm python sample.py \
  --tore-type SDTM_testidm2 \
  --prompt "A red bicycle beside a canal at sunrise" \
  --prompt "A watercolor painting of a snow-covered village" \
  --output-path ./samples
```

Each `--prompt` generates one image. Use either `--prompt` or `--dataset` in
the same command. `sample_testidm2.py` is a compatibility entry point that
defaults to `SDTM_testidm2`.

### Reported metrics

- `seconds_per_image` is the synchronized pipeline time divided by `timed_images`.
  It includes text encoding, denoising, and decoding, with warmup, image saving,
  FLOP profiling, and evaluation measured outside that interval. Sampling
  resumes by default. A partial run times the newly generated images. If all
  images already exist, it times the first batch again without rewriting them.
- `transformer_parameters` counts the parameters after applying the selected
  method. Token merging reduces the sequence length during inference and keeps
  the same FLUX transformer weights.
- `transformer_flops_per_image` reports PyTorch-counted FLOPs for the FLUX
  transformer over all denoising steps of the first input batch, divided by its
  image count. The separate profiling pass uses the same prompts and seeds.
  The manifest records the profiled IDs, operator counts, and per-step FLOPs.
  The counter covers supported operations such as matrix multiplication and
  attention matmuls, with two FLOPs per multiply-add. Text encoders, VAE, and
  operations such as indexing and normalization are outside this count.
- Dataset runs also report `clip_score`, `fid`, and `evaluated_images`. CLIP
  Score uses OpenCLIP ViT-g/14 and FID uses Clean-FID with the local Inception
  weights. `--metric-device` and `--metric-batch-size` control evaluation.

FLUX.1-dev is loaded as `bfloat16` by default. Use `--cpu-offload` on a CUDA
machine that cannot hold the complete pipeline in GPU memory. Offload affects
latency and therefore should be identical across modes in timing comparisons.

## Standalone FLOP profiling

Use `measure_flops.py` when you only need the Transformer FLOP report.

```bash
conda run -n sdtm python measure_flops.py \
  --modes Default,ToMe,SDTM,SDTM_testidm2,SDTM_TaylorSeer \
  --height 1024 --width 1024 --num-inference-steps 50 \
  --batch-size 1 --output-json flops_flux.json
```

## Standalone CLIP Score and FID

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
