import torch
import os
from cleanfid import fid
from cleanfid.inception_torchscript import InceptionV3W

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"  

# 加载inception-v3模型
class CustomInceptionExtractor(torch.nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        path = "inceptions"  ### 写绝对路径
        self.model = InceptionV3W(path, download=False).to(device)

    def forward(self, x):
        return self.model(x)

custom_extractor = CustomInceptionExtractor(device='cuda')


def compute_fid(real_path, fake_path):
    """
    Compute FID between real and fake images.
    """
    score = fid.compute_fid(real_path, fake_path, custom_feat_extractor=custom_extractor)
    return score

if __name__ == "__main__":
    real_images_path = "datasets/val2017"
    sdtm_images_path = "samples/SD3M-SDTM-R0.3-D0.2-Sw20-rnd1-2x2-as0.05-ad0.05-ap2-PmM-W0.1-Ps3-Pl-1-CESTrue-1024x1024-steps50-cfg7.0-seed0"
    default_images_path = "../../../irip_16t_0/huangyu_2026/samples/COCO2017/Default"
    tome_images_path = "../../../irip_16t_0/huangyu_2026/samples/COCO2017/ToMe"
    mine_images_path = "../../../irip_16t_0/huangyu_2026/samples/COCO2017/SSM_Modify_Version1.0"
    sdtm_taylorseer_images_path = "../../../irip_16t_0/huangyu_2026/samples/COCO2017/SDTM_TaylorSeer"
    default_fid_score = compute_fid(real_images_path, default_images_path)
    sdtm_fid_score = compute_fid(real_images_path, sdtm_images_path)
    tome_fid_score = compute_fid(real_images_path, tome_images_path)
    mine_fid_score = compute_fid(real_images_path, mine_images_path)
    sdtm_taylorseer_fid_score = compute_fid(real_images_path, sdtm_taylorseer_images_path)
    print(f"Default FID Score: {default_fid_score}")
    print(f"SDTM FID Score: {sdtm_fid_score}")
    print(f"ToMe FID Score: {tome_fid_score}")
    print(f"Mine FID Score: {mine_fid_score}")
    print(f"SDTM TaylorSeer FID Score: {sdtm_taylorseer_fid_score}")