"""
CLIP 分数评估脚本

计算生成图像集合相对于 COCO val2017 字幕的 CLIP 分数（图文相似度）。

使用方式:
    1. 先在有网络的环境运行 download_clip_model.py 下载模型到本地
    2. 再运行本脚本计算 CLIP 分数:
       python clip_evaluation.py

CLIP Score 定义:
    对每张生成图像，计算其与对应文本描述的余弦相似度，
    取 mean(max(100 * cosine_similarity, 0)) 作为最终得分。
    分数越高代表图文一致性越好（最高约 30~35）。
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import open_clip

# 抑制 tokenizer 并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ──────────────────────────────────────────────
# 字幕加载
# ──────────────────────────────────────────────

def load_coco_captions(captions_file):
    """
    从 COCO 字幕文件中加载 {image_id -> 最长字幕} 的映射。
    文件格式为扁平列表: [{"image_id": int, "caption": str}, ...]
    """
    with open(captions_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    captions_dict = {}
    for item in data:
        image_id = item["image_id"]
        caption = item["caption"]
        # 每个 image_id 保留最长字幕（与生成图像时使用的策略一致）
        if image_id not in captions_dict or len(caption) > len(captions_dict[image_id]):
            captions_dict[image_id] = caption

    print(f"共加载 {len(captions_dict)} 个图像对应的字幕（每图取最长字幕）")
    return captions_dict


# ──────────────────────────────────────────────
# CLIP 评估器
# ──────────────────────────────────────────────

class CLIPScoreEvaluator:
    """
    使用本地 open_clip 模型计算 CLIP Score。
    在初始化时从本地路径加载模型权重，无需网络连接。
    """

    def __init__(
        self,
        model_path: str = "clip_models/ViT-L-14-openai.pt",
        model_name: str = "ViT-L-14",
        device: str = "cuda",
    ):
        self.device = device

        print(f"正在从本地加载 CLIP 模型: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"CLIP 模型文件不存在: {model_path}\n"
                "请先在有网络的环境运行 download_clip_model.py 下载模型。"
            )

        # 创建模型架构（不下载预训练权重）
        model, _, preprocess = open_clip.create_model_and_transforms(model_name)

        # 加载本地保存的权重
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

        self.model = model.to(device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)
        print("CLIP 模型加载完成。")

    @torch.no_grad()
    def compute_clip_score(
        self,
        images_path: str,
        captions_dict: dict,
        batch_size: int = 32,
    ) -> float:
        """
        计算指定图像目录中所有图像的 CLIP Score。

        Args:
            images_path:   图像目录路径
            captions_dict: {image_id (int) -> caption (str)} 映射
            batch_size:    每批处理的图像数量

        Returns:
            平均 CLIP Score（标量）
        """
        image_files = sorted([
            f for f in os.listdir(images_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if not image_files:
            print(f"  警告: 目录中无图像文件: {images_path}")
            return 0.0

        all_scores = []
        skipped = 0

        for start in tqdm(
            range(0, len(image_files), batch_size),
            desc=f"  {os.path.basename(images_path)}",
        ):
            batch_files = image_files[start: start + batch_size]
            batch_images = []
            batch_texts = []

            for fname in batch_files:
                # 文件名去掉扩展名后转为整数即为 image_id
                try:
                    image_id = int(os.path.splitext(fname)[0])
                except ValueError:
                    skipped += 1
                    continue

                if image_id not in captions_dict:
                    skipped += 1
                    continue

                img_path = os.path.join(images_path, fname)
                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception:
                    skipped += 1
                    continue

                batch_images.append(self.preprocess(image))
                batch_texts.append(captions_dict[image_id])

            if not batch_images:
                continue

            images_tensor = torch.stack(batch_images).to(self.device)
            texts_tokens = self.tokenizer(batch_texts).to(self.device)

            image_features = self.model.encode_image(images_tensor)
            text_features = self.model.encode_text(texts_tokens)

            # L2 归一化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 余弦相似度，缩放到 0~100，截断负值
            cosine_sim = (image_features * text_features).sum(dim=-1)
            scores = torch.clamp(100.0 * cosine_sim, min=0.0).cpu().numpy()
            all_scores.extend(scores.tolist())

        if skipped > 0:
            print(f"  注意: {skipped} 张图像因找不到对应字幕或读取失败而跳过")

        if not all_scores:
            return 0.0

        return float(np.mean(all_scores))


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # ── 路径配置 ──────────────────────────────
    CAPTIONS_FILE = "datasets/COCO2017/captions_val2017.json"
    CLIP_MODEL_PATH = "clip_models/ViT-L-14-openai.pt"

    # 三个待评估的图像集合（名称 -> 路径）
    IMAGE_SETS = {
        "Default": (
            "../../../irip_16t_0/huangyu_2026/samples/COCO2017/"
            "Default"
        ),
        "ToMe": (
            "../../../irip_16t_0/huangyu_2026/samples/COCO2017/"
            "ToMe"
        ),
        "SDTM": (
            "samples/SD3M-SDTM-R0.3-D0.2-Sw20-rnd1-2x2-as0.05-ad0.05-ap2-PmM-"
            "W0.1-Ps3-Pl-1-CESTrue-1024x1024-steps50-cfg7.0-seed0"
        ),
        "Mine": (
            "../../../irip_16t_0/huangyu_2026/samples/COCO2017/"
            "SSM_Modify_Version1.0"
        ),
        "SDTM_TaylorSeer": (
            "../../../irip_16t_0/huangyu_2026/samples/COCO2017/"
            "SDTM_TaylorSeer"
        )
    }

    BATCH_SIZE = 32
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # ─────────────────────────────────────────

    print(f"使用设备: {DEVICE}")
    print(f"\n正在加载 COCO 字幕: {CAPTIONS_FILE}")
    captions_dict = load_coco_captions(CAPTIONS_FILE)

    evaluator = CLIPScoreEvaluator(
        model_path=CLIP_MODEL_PATH,
        model_name="ViT-L-14",
        device=DEVICE,
    )

    results = {}
    print("\n===== 开始计算 CLIP Score =====")

    for name, path in IMAGE_SETS.items():
        print(f"\n[{name}] 图像路径: {path}")
        if not os.path.exists(path):
            print(f"  路径不存在，跳过。")
            continue
        score = evaluator.compute_clip_score(path, captions_dict, batch_size=BATCH_SIZE)
        results[name] = score
        print(f"  {name} CLIP Score: {score:.4f}")

    print("\n===== 最终结果汇总 =====")
    for name, score in results.items():
        print(f"  {name:10s}  CLIP Score: {score:.4f}")
