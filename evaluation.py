import json
import argparse
import torch
import os
import time
from tqdm import tqdm
from cleanfid import fid
from cleanfid.inception_torchscript import InceptionV3W
from torchmetrics.image.inception import InceptionScore

from TR_SDTM import apply_SDTM
from diffusers import StableDiffusion3Pipeline

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



def load_captions(file_path, num_samples=100):
    """
    Load captions and limit to num_samples for evaluation.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    annotations = data
    captions = []
    seen_ids = set()

    for item in annotations:
        image_id = item['image_id']
        if image_id not in seen_ids:
            captions.append({'image_id': image_id, 'caption': item['caption']})
            seen_ids.add(image_id)
        if len(captions) >= num_samples:
            break

    return captions

def generate_images(pipe, captions_list, args, output_path, model_type):
    """
    Generate images using the pipeline.
    """
    batch_size = args.batch_size
    generator = torch.Generator(device=pipe.device).manual_seed(args.seed)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    start_time = time.time()

    for i in tqdm(range(0, len(captions_list), batch_size), desc=f"Generating {model_type} images"):
        batch_captions = captions_list[i: i + batch_size]
        prompt_list = [item['caption'] for item in batch_captions]
        id_list = [item['image_id'] for item in batch_captions]

        images = pipe(
            prompt=prompt_list,
            generator=generator,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        ).images

        for j, image in enumerate(images):
            image_id = str(id_list[j]).zfill(12)
            image.save(os.path.join(output_path, f"{image_id}.jpg"))

    end_time = time.time()
    generation_time = end_time - start_time

    return generation_time

def compute_fid(real_path, fake_path):
    """
    Compute FID between real and fake images.
    """
    score = fid.compute_fid(real_path, fake_path, custom_feat_extractor=custom_extractor)
    return score

def compute_is(image_path):
    """
    Compute Inception Score for generated images.
    """
    from PIL import Image
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    images = []
    for img_file in os.listdir(image_path):
        if img_file.endswith('.jpg') or img_file.endswith('.png'):
            img = Image.open(os.path.join(image_path, img_file)).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)

    if not images:
        raise ValueError("No images found in the directory")

    images = torch.stack(images)

    inception = InceptionScore()
    inception.update(images)
    score = inception.compute()
    return score

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load captions
    captions_list = load_captions(args.caption_path, args.num_samples)
    print(f"Loaded {len(captions_list)} captions for evaluation.")

    # Load pipeline
    if args.torch_dtype == "float32":
        pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=torch.float32)
    elif args.torch_dtype == "float16":
        pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)

    pipe = pipe.to(device)

    # Extract model name
    model_name = os.path.basename(args.model_path)
    if "stable-diffusion-3-medium-diffusers" in model_name:
        model_name = "SD3M"

    # Output directories
    default_output = os.path.join(args.output_path, f"{model_name}-Default-eval")
    sdtm_output = os.path.join(args.output_path, f"{model_name}-SDTM-eval")

    # Generate Default images
    print("Generating Default images...")
    default_time = generate_images(pipe, captions_list, args, default_output, "Default")

    # Apply SDTM
    print("Applying SDTM...")
    pipe = apply_SDTM(
        pipe,
        ratio=args.SDTM_ratio,
        deviation=args.SDTM_deviation,
        switch_step=args.SDTM_switch_step,
        use_rand=args.SDTM_use_rand,
        sx=args.SDTM_sx,
        sy=args.SDTM_sy,
        a_s=args.SDTM_a_s,
        a_d=args.SDTM_a_d,
        a_p=args.SDTM_a_p,
        pseudo_merge=args.SDTM_pseudo_merge,
        mcw=args.SDTM_mcw,
        protect_steps_frequency=args.SDTM_protect_steps_frequency,
        protect_layers_frequency=args.SDTM_protect_layers_frequency,
    )
    print(f"SDTM applied. Pipe type: {type(pipe)}")
    if hasattr(pipe, '_tore_info'):
        print("SDTM info present.")
    else:
        print("SDTM info NOT present!")

    # Generate SDTM images
    print("Generating SDTM images...")
    sdtm_time = generate_images(pipe, captions_list, args, sdtm_output, "SDTM")

    # Compute metrics
    print("Computing metrics...")

    # FID with COCO val as reference
    fid_default = compute_fid(sdtm_output, default_output)

    '''
    fid_sdtm = compute_fid('coco-val', sdtm_output)

    # IS
    is_default = compute_is(default_output)
    is_sdtm = compute_is(sdtm_output)
    '''

    # Speedup ratio
    speedup = default_time / sdtm_time if sdtm_time > 0 else float('inf')

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Number of samples: {len(captions_list)}")
    print(f"Image size: {args.height}x{args.width}")
    print(f"Inference steps: {args.num_inference_steps}")
    print()
    print("TIMING:")
    print(f"Default generation time: {default_time:.2f}")
    print(f"SDTM generation time: {sdtm_time:.2f}")
    print(f"Speedup ratio: {speedup:.2f}")
    print()
    print("FID (lower is better):")
    print(f"Default FID: {fid_default:.4f}")
    # print(f"SDTM FID: {fid_sdtm:.4f}")
    # print()
    # print("Inception Score (higher is better):")
    # print(f"Default IS: {is_default:.4f}")
    # print(f"SDTM IS: {is_sdtm:.4f}")
    # print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Default vs SDTM models")
    parser.add_argument("--caption-path", type=str, default='datasets/COCO2017/captions_val2017.json')
    parser.add_argument("--output-path", type=str, default="evaluation_results")
    parser.add_argument("--model-path", type=str, default="checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--torch-dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--batch-size", type=int, default=1)  # Smaller batch for evaluation
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=10)  # Number of samples for evaluation

    # SDTM arguments (same as sample.py)
    parser.add_argument("--SDTM-ratio", type=float, default=0.3)
    parser.add_argument("--SDTM-deviation", type=float, default=0.2)
    parser.add_argument("--SDTM-switch-step", type=int, default=20)
    parser.add_argument("--SDTM-use-rand", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--SDTM-sx", type=int, default=2)
    parser.add_argument("--SDTM-sy", type=int, default=2)
    parser.add_argument("--SDTM-a-s", type=float, default=0.05)
    parser.add_argument("--SDTM-a-d", type=float, default=0.05)
    parser.add_argument("--SDTM-a-p", type=float, default=2)
    parser.add_argument("--SDTM-pseudo-merge", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--SDTM-mcw", type=float, default=0.1)
    parser.add_argument("--SDTM-protect-steps-frequency", type=int, default=1)
    parser.add_argument("--SDTM-protect-layers-frequency", type=int, default=1)
    parser.add_argument("--SDTM-cache-each-step", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()
    main(args)