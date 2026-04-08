"""
PartiPrompts CLIP Score 评估脚本

计算生成图像集合相对于 PartiPrompts 的 CLIP Score
"""

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import open_clip
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ──────────────────────────────────────────────
# PartiPrompts 加载
# ──────────────────────────────────────────────

def load_partiprompts(tsv_file):
    """
    加载 PartiPrompts.tsv

    返回:
        {index -> prompt}
    """
    df = pd.read_csv(tsv_file, sep="\t")

    prompts = {}
    for i, row in df.iterrows():
        prompts[i] = row["Prompt"]

    print(f"共加载 {len(prompts)} 条 PartiPrompts")
    return prompts


# ──────────────────────────────────────────────
# CLIP 评估器
# ──────────────────────────────────────────────

class CLIPScoreEvaluator:

    def __init__(
        self,
        model_path: str = "../clip_models/ViT-L-14-openai.pt",
        model_name: str = "ViT-L-14",
        device: str = "cuda",
    ):
        self.device = device

        print(f"正在从本地加载 CLIP 模型: {model_path}")

        model, _, preprocess = open_clip.create_model_and_transforms(model_name)

        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

        self.model = model.to(device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)

        print("CLIP 模型加载完成")

    @torch.no_grad()
    def compute_clip_score(
        self,
        images_path: str,
        prompts_dict: dict,
        batch_size: int = 32,
    ):

        image_files = sorted([
            f for f in os.listdir(images_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        all_scores = []
        skipped = 0

        for start in tqdm(range(0, len(image_files), batch_size)):

            batch_files = image_files[start:start + batch_size]

            batch_images = []
            batch_texts = []

            for fname in batch_files:

                try:
                    idx = int(os.path.splitext(fname)[0])
                except:
                    skipped += 1
                    continue

                if idx not in prompts_dict:
                    skipped += 1
                    continue

                img_path = os.path.join(images_path, fname)

                try:
                    image = Image.open(img_path).convert("RGB")
                except:
                    skipped += 1
                    continue

                batch_images.append(self.preprocess(image))
                batch_texts.append(prompts_dict[idx])

            if not batch_images:
                continue

            images = torch.stack(batch_images).to(self.device)
            texts = self.tokenizer(batch_texts).to(self.device)

            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(texts)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            cosine_sim = (image_features * text_features).sum(dim=-1)

            scores = torch.clamp(100 * cosine_sim, min=0).cpu().numpy()

            all_scores.extend(scores.tolist())

        if skipped > 0:
            print(f"跳过 {skipped} 张图像")

        return float(np.mean(all_scores))


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────

if __name__ == "__main__":

    PARTI_FILE = "../datasets/PartiPrompts/PartiPrompts.tsv"
    CLIP_MODEL_PATH = "../clip_models/ViT-L-14-openai.pt"

    IMAGE_SETS = {
        "Default": (
            "../../../../irip_16t_0/huangyu_2026/samples/"
            "Default"
        ),
        "ToMe": (
            "../../../../irip_16t_0/huangyu_2026/samples/"
            "ToMe"
        ),
        "SDTM": (
            "../samples/SD3M-SDTM-R0.3-D0.2-Sw20-rnd1-2x2-as0.05-ad0.05-ap2-PmM-"
            "W0.1-Ps3-Pl-1-CESTrue-1024x1024-steps50-cfg7.0-seed0"
        ),
        "Mine": (
            "../../../../irip_16t_0/huangyu_2026/samples/"
            "SSM_Modify_Version1.0"
        )
    }
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("加载 PartiPrompts...")
    prompts_dict = load_partiprompts(PARTI_FILE)

    evaluator = CLIPScoreEvaluator(
        model_path=CLIP_MODEL_PATH,
        model_name="ViT-L-14",
        device=DEVICE,
    )

    print("\n===== 开始评估 =====")

    results = {}

    for name, path in IMAGE_SETS.items():

        print(f"\n{name}")

        if not os.path.exists(path):
            print("路径不存在")
            continue

        score = evaluator.compute_clip_score(path, prompts_dict)

        results[name] = score

        print(f"{name} CLIP Score: {score:.4f}")

    print("\n===== 最终结果 =====")

    for k, v in results.items():
        print(f"{k:10s}: {v:.4f}")