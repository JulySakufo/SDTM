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

# def compute_fid(fake_path):
#     return fid.compute_fid(
#         fake_path,
#         dataset_name="coco2017",
#         dataset_split="val",
#         custom_feat_extractor=custom_extractor
#     )


if __name__ == "__main__":
    real_images_path = "datasets/val2017_center_crop_1024"
    # sdtm_images_path = "samples/SD3M-SDTM-R0.3-D0.2-Sw20-rnd1-2x2-as0.05-ad0.05-ap2-PmM-W0.1-Ps3-Pl-1-CESTrue-1024x1024-steps50-cfg7.0-seed0"
    default_images_path = "../../../irip_16t_0/huangyu_2026/samples/COCO2017/Default"
    # tome_images_path = "../../../irip_16t_0/huangyu_2026/samples/COCO2017/ToMe"
    # mine_images_path = "../../../irip_16t_0/huangyu_2026/samples/COCO2017/SSM_Modify_Version1.0"
    # mine2_images_path = "../../../irip_16t_0/huangyu_2026/samples/COCO2017/SSM+IDM_Version1.0"
    # sdtm_taylorseer_images_path = "../../../irip_16t_0/huangyu_2026/samples/COCO2017/SDTM_TaylorSeer"
    # sdtm2_taylorseer_images_path = "../../../irip_16t_0/huangyu_2026/samples/COCO2017/SDTM2_TaylorSeer"
    # sdtm_taylorseer_1_1_5_images_path = "../../../irip_16t_0/huangyu_2026/samples/COCO2017/SDTM_TaylorSeer_1_1_5"
    sdtm_taylorseer_2_1_12_images_path = "../../../irip_16t_0/huangyu_2026/samples/COCO2017/SDTM_TaylorSeer_2_1_12"
    # sdtm_taylorseer_3_1_5_images_path = "../../../irip_16t_0/huangyu_2026/samples/COCO2017/SDTM_TaylorSeer_3_1_5"
    # sdtm_taylorseer_4_1_3_images_path = "../../../irip_16t_0/huangyu_2026/samples/COCO2017/SDTM_TaylorSeer_4_1_3"
    sdtm_taylorseer_4_2_2_images_path = "../../../irip_16t_0/huangyu_2026/samples/COCO2017/SDTM_TaylorSeer_4_2_2"
    sdtm_taylorseer_4_2_4_images_path = "../../../irip_16t_0/huangyu_2026/samples/COCO2017/SDTM_TaylorSeer_4_2_4"
    taylorseer_only_4_2_2_images_path = "../../../irip_16t_0/huangyu_2026/samples/COCO2017/TaylorSeer_Only_4_2_2"
    taylorseer_only_2_1_12_images_path = "../../../irip_16t_0/huangyu_2026/samples/COCO2017/TaylorSeer_Only_2_1_12"
    default_fid_score = compute_fid(real_images_path, default_images_path)
    # sdtm_fid_score = compute_fid(real_images_path, sdtm_images_path)
    # tome_fid_score = compute_fid(real_images_path, tome_images_path)
    # mine_fid_score = compute_fid(real_images_path, mine_images_path)
    # mine2_fid_score = compute_fid(real_images_path, mine2_images_path)
    # sdtm_taylorseer_fid_score = compute_fid(real_images_path, sdtm_taylorseer_images_path)
    # sdtm2_taylorseer_fid_score = compute_fid(real_images_path, sdtm2_taylorseer_images_path)
    # sdtm_taylorseer_1_1_5_fid_score = compute_fid(real_images_path, sdtm_taylorseer_1_1_5_images_path)
    sdtm_taylorseer_2_1_12_fid_score = compute_fid(real_images_path, sdtm_taylorseer_2_1_12_images_path)
    # sdtm_taylorseer_3_1_5_fid_score = compute_fid(real_images_path, sdtm_taylorseer_3_1_5_images_path)
    # sdtm_taylorseer_4_1_3_fid_score = compute_fid(real_images_path, sdtm_taylorseer_4_1_3_images_path)
    sdtm_taylorseer_4_2_2_fid_score = compute_fid(real_images_path, sdtm_taylorseer_4_2_2_images_path)
    sdtm_taylorseer_4_2_4_fid_score = compute_fid(real_images_path, sdtm_taylorseer_4_2_4_images_path)
    taylorseer_only_4_2_2_fid_score = compute_fid(real_images_path, taylorseer_only_4_2_2_images_path)
    taylorseer_only_2_1_12_fid_score = compute_fid(real_images_path, taylorseer_only_2_1_12_images_path)
    print(f"Default FID Score: {default_fid_score}")
    # print(f"SDTM FID Score: {sdtm_fid_score}")
    # print(f"ToMe FID Score: {tome_fid_score}")
    # print(f"Mine FID Score: {mine_fid_score}")
    # print(f"Mine2 FID Score: {mine2_fid_score}")
    # print(f"SDTM TaylorSeer FID Score: {sdtm_taylorseer_fid_score}")
    # print(f"SDTM2 TaylorSeer FID Score: {sdtm2_taylorseer_fid_score}")
    # print(f"SDTM TaylorSeer 1 1 5 FID Score: {sdtm_taylorseer_1_1_5_fid_score}")
    print(f"SDTM TaylorSeer 2 1 12 FID Score: {sdtm_taylorseer_2_1_12_fid_score}")
    # print(f"SDTM TaylorSeer 3 1 5 FID Score: {sdtm_taylorseer_3_1_5_fid_score}")
    # print(f"SDTM TaylorSeer 4 1 3 FID Score: {sdtm_taylorseer_4_1_3_fid_score}")
    print(f"SDTM TaylorSeer 4 2 2 FID Score: {sdtm_taylorseer_4_2_2_fid_score}")
    print(f"SDTM TaylorSeer 4 2 4 FID Score: {sdtm_taylorseer_4_2_4_fid_score}")
    print(f"TaylorSeer Only 4 2 2 FID Score: {taylorseer_only_4_2_2_fid_score}")
    print(f"TaylorSeer Only 2 1 12 FID Score: {taylorseer_only_2_1_12_fid_score}")