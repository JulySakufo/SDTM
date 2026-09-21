# SDTM for Qwen-Image-Edit

本目录把父目录中已经验证的 ToMe、SDTM、SDTM_testidm2 和 SDTM+TaylorSeer 迁移到原始单图版 `Qwen/Qwen-Image-Edit`。实现只在运行时 patch pipeline 实例，不修改父目录实现、Diffusers 源码或本地模型权重。

Qwen 的 image stream 是 `[target noisy latent, reference-image latent]` 的拼接。本实现默认只合并 target；`--token-scope all-separate` 会分别合并两段，但永远不跨段，文本 token 始终完整。不能组成完整窗口的底边和右边 token 会被永久保留。

## 环境

```bash
cd /irip/huangyu_2026/SDTM/SDTM-Qwen_image_edit
conda env create -f environment.yml
conda activate sdtm-qwen-image-edit
python -m unittest discover -s tests -v
```

环境独立固定为 Python 3.10、Diffusers 0.35.1、Transformers 4.55.2、PyTorch 2.4.1/CUDA 11.8，不会改变现有 `sdtm` 环境。这里必须使用 Transformers 4.55.2：本地 checkpoint 的 Qwen2.5-VL 配置包含新版嵌套 `text_config`，4.51.x 无法正确反序列化。默认读取：

代码会为 Diffusers 0.35.1 自动安装一个运行时 direct-SDPA processor。原因是该版本 Diffusers 会无条件向 SDPA 传递 `enable_gqa=False`，而 PyTorch 2.4 尚不接受此参数；Qwen image attention 的 Q/KV head 数相同，省略这个恒为 false 的参数与官方 native 路径等价，无需升级或修改已安装的 PyTorch/Diffusers。

- 模型：`/irip_16t_0/huangyu_2026/models/Qwen-Image-Edit`
- COCO 原图：`../datasets/val2017`
- 5,000 条最长 caption：`../datasets/COCO2017/captions_val2017.json`
- 输出：`/irip_16t_0/huangyu_2026/samples`

如机器驱动使用另一 CUDA toolkit，只需在新环境中替换 `pytorch-cuda`，不要更改 Diffusers/Transformers 固定版本。

## 五种采样模式

默认参数遵循官方质量配置：BF16、50 steps、`true_cfg_scale=4.0`、空白 negative prompt、约 1MP 的输入长宽比、batch size 1、seed 0。caption 直接作为编辑 prompt，不做 rewriting。

```bash
# 原生 Qwen-Image-Edit
python sample.py --tore-type Default

# ToMe
python sample.py --tore-type ToMe --ToMe-ratio 0.9

# SDTM
python sample.py --tore-type SDTM --SDTM-ratio 0.3 --SDTM-deviation 0.2

# SDTM_testidm2（也可使用 python sample_testidm2.py）
python sample.py --tore-type SDTM_testidm2

# SDTM_testidm2 + TaylorSeer
python sample.py --tore-type SDTM_TaylorSeer
```

小规模 smoke test 使用 `--limit`，固定输出尺寸时必须同时给出高和宽：

```bash
python sample.py --tore-type SDTM_testidm2 --limit 2 --warmup-runs 0
python sample.py --tore-type ToMe --token-scope all-separate --height 1024 --width 1024 --limit 2
```

结果按 `{image_id:012d}.jpg` 保存。再次执行默认跳过已有图片，并原子更新 `manifest.json`。manifest 包含实际 target/reference 尺寸、CFG 调用配置、patch/token 统计、参数、耗时和软件版本。相同 `image_id` 在所有模式中使用相同 latent seed；合并随机序列也由该 seed、denoising step、layer 和 segment 稳定派生。

ToMe 的 `ratio` 沿用父目录语义，是保留比例；SDTM 的 `ratio` 是随余弦调度变化的缩减控制量。`--token-scope target` 是推荐且默认的比较口径。

## FLOPs

只统计 `QwenImageTransformer2DModel`，文本编码器和 VAE 不计入。true CFG 默认让每个 denoising step 对应 cond/uncond 两次 transformer 调用，因此 50 steps 应记录 100 calls。

```bash
python measure_flops.py \
  --modes Default,ToMe,SDTM,SDTM_testidm2,SDTM_TaylorSeer \
  --flops-image-id 139
```

输出同时报告每次 transformer call、每个 denoising step、每张编辑图的 FLOPs，cond/uncond、Taylor/full 与算子分解，以及相对 Default 的降幅。检查零缩减对照可运行：

```bash
python measure_flops.py --modes Default,SDTM --SDTM-ratio 0 --SDTM-deviation 0
```

## CLIP Score 与 FID

小样本只用于验证评测链路：

```bash
python clip_evaluation.py \
  --image-set smoke=/path/to/smoke/images \
  --expected-count 2 --allow-incomplete \
  --output-json clip_smoke.json

python fid_evaluation.py \
  --image-set smoke=/path/to/smoke/images \
  --expected-count 2 --allow-incomplete \
  --output-json fid_smoke.json
```

正式 5K 评测默认严格检查全部 COCO image id 和损坏文件：

```bash
python clip_evaluation.py \
  --image-set Default=/path/to/Default \
  --image-set ToMe=/path/to/ToMe \
  --image-set SDTM=/path/to/SDTM \
  --image-set SDTM_testidm2=/path/to/SDTM_testidm2 \
  --image-set SDTM_TaylorSeer=/path/to/SDTM_TaylorSeer \
  --output-json clip_scores_qwen_5k.json

python fid_evaluation.py \
  --image-set Default=/path/to/Default \
  --image-set ToMe=/path/to/ToMe \
  --image-set SDTM=/path/to/SDTM \
  --image-set SDTM_testidm2=/path/to/SDTM_testidm2 \
  --image-set SDTM_TaylorSeer=/path/to/SDTM_TaylorSeer \
  --output-json fid_scores_qwen_5k.json
```

CLIP 延续 OpenCLIP ViT-G/14 的生成图–COCO caption 分数；FID 延续 clean-fid，并与 `../datasets/val2017_center_crop_1024` 比较。

## 接口与限制

四个 wrapper 保留 `apply_ToMe(...)`、`apply_SDTM(...)`、`apply_SDTM_TaylorSeer(...)`，并统一增加 `token_scope="target"`；同时公开 `remove_patch`、`set_batch_seeds`、`patch_statistics`。当前明确不支持 2509/2511 Plus、多参考图、ControlNet、LoRA 或量化模型。

此协议适合比较 Qwen Default 与四个加速模式的相对质量/FLOPs。Qwen 额外看到了对应 COCO 原图，因此绝对 CLIP/FID 不应和只接收文本的 SD3/FLUX 作公平横向排名。
