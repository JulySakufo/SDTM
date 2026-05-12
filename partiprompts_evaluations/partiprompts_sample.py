import csv
import argparse
import torch
import os

from tqdm import tqdm

from TR_ToMe import apply_ToMe
from TR_SDTM import apply_SDTM
from TR_SDTM_TaylorSeer import apply_SDTM_TaylorSeer

from diffusers import StableDiffusion3Pipeline

def load_prompts(file_path):
    """
    Reads the PartiPrompts TSV file and extracts the prompts along with their metadata.

    :param file_path: Path to the PartiPrompts TSV file
    :return: A list where each element is a dictionary containing 'prompt_id', 'prompt', 'category', 'challenge', and 'note'
    """
    prompts_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for idx, row in enumerate(reader):
            prompts_list.append({
                'prompt_id': idx,
                'prompt': row['Prompt'],
                'category': row['Category'],
                'challenge': row['Challenge'],
                'note': row.get('Note', ''),
            })
    return prompts_list

def main(args):
    """
    Run sampling on single GPU.
    """
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    file_path = args.caption_path
    prompts_list = load_prompts(file_path)

    print(f"Loaded {len(prompts_list)} prompts from PartiPrompts.")

    # Load the pipeline
    if args.torch_dtype == "float32":
        pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=torch.float32)
    elif args.torch_dtype == "float16":
        pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
    
    pipe = pipe.to(device)

    # Construct output path
    # Extract and simplify model name
    model_name = os.path.basename(args.model_path)
    if "stable-diffusion-3-medium-diffusers" in model_name:
        model_name = "SD3M"
    
    batch_size = args.batch_size
    generator = torch.Generator(device=device).manual_seed(args.seed)

    if args.tore_type is None or args.tore_type == "Default":
        output_path = os.path.join(
            args.output_path,
            f"{model_name}-Default-{args.height}x{args.width}-steps{args.num_inference_steps}-cfg{args.guidance_scale}-seed{args.seed}"
        )
        pipe = pipe

    elif args.tore_type == "ToMe":
        output_path = os.path.join(
            args.output_path,
            f"{model_name}-ToMe-pseudo_merge{args.ToMe_pseudo_merge}-{args.ToMe_ratio}-{args.ToMe_sx}x{args.ToMe_sy}-"
            f"MergeAttn{args.ToMe_merge_attn}-MergeMLP{args.ToMe_merge_mlp}-"
            f"{args.height}x{args.width}-steps{args.num_inference_steps}-cfg{args.guidance_scale}-seed{args.seed}"
        )
        output_path = (
            output_path
            .replace("pseudo_mergeFalse", "Merge").replace("pseudo_mergeTrue", "PseudoMerge")
            .replace("MergeAttnTrue", "MergeAttn").replace("MergeAttnFalse", "UnMergeAttn")
            .replace("MergeMLPTrue", "MergeMLP").replace("MergeMLPFalse", "UnMergeMLP")
        )
        apply_ToMe(
            pipe, 
            ratio = args.ToMe_ratio, 
            sx = args.ToMe_sx, 
            sy = args.ToMe_sy, 
            use_rand = args.ToMe_use_rand, 
            merge_attn = args.ToMe_merge_attn, 
            merge_mlp = args.ToMe_merge_mlp,
            change_merge_to_prune = args.ToMe_pseudo_merge
            # pseudo_merge = args.ToMe_pseudo_merge
        )

    elif args.tore_type == "SDTM":
            # Compact, fully-parameterized SDTM tag covering all related args
            sdtm_tag = (
                f"{model_name}-SDTM-"
                f"R{args.SDTM_ratio:g}-D{args.SDTM_deviation:g}-Sw{args.SDTM_switch_step}-"
                f"rnd{int(args.SDTM_use_rand)}-{args.SDTM_sx}x{args.SDTM_sy}-"
                f"as{args.SDTM_a_s:g}-ad{args.SDTM_a_d:g}-ap{args.SDTM_a_p:g}-"
                f"Pm{'PM' if args.SDTM_pseudo_merge else 'M'}-W{args.SDTM_mcw:g}-"
                f"Ps{args.SDTM_protect_steps_frequency}-Pl{args.SDTM_protect_layers_frequency}-CES{args.SDTM_cache_each_step}"
            )
            output_path = os.path.join(
                args.output_path,
                f"{sdtm_tag}-{args.height}x{args.width}-steps{args.num_inference_steps}-cfg{args.guidance_scale}-seed{args.seed}"
            )
            apply_SDTM(
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

    elif args.tore_type == "SDTM_TaylorSeer":
            sdtm_ts_tag = (
                f"{model_name}-SDTM_TaylorSeer-"
                f"R{args.SDTM_ratio:g}-D{args.SDTM_deviation:g}-Sw{args.SDTM_switch_step}-"
                f"rnd{int(args.SDTM_use_rand)}-{args.SDTM_sx}x{args.SDTM_sy}-"
                f"as{args.SDTM_a_s:g}-ad{args.SDTM_a_d:g}-ap{args.SDTM_a_p:g}-"
                f"Pm{'PM' if args.SDTM_pseudo_merge else 'M'}-W{args.SDTM_mcw:g}-"
                f"Ps{args.SDTM_protect_steps_frequency}-Pl{args.SDTM_protect_layers_frequency}-"
                f"TI{args.Taylor_interval}-TO{args.Taylor_max_order}-TE{args.Taylor_first_enhance}"
            )
            output_path = os.path.join(
                args.output_path,
                f"{sdtm_ts_tag}-{args.height}x{args.width}-steps{args.num_inference_steps}-cfg{args.guidance_scale}-seed{args.seed}"
            )
            apply_SDTM_TaylorSeer(
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
                taylor_interval=args.Taylor_interval,
                taylor_max_order=args.Taylor_max_order,
                taylor_first_enhance=args.Taylor_first_enhance,
            )

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print(f"Output path: {output_path}")

    for i in tqdm(range(0, len(prompts_list), batch_size), desc="Generating images"):
        batch_prompts = prompts_list[i: i + batch_size]
        prompt_list = [item['prompt'] for item in batch_prompts]
        id_list = [item['prompt_id'] for item in batch_prompts]

        images = pipe(
            prompt=prompt_list,
            generator=generator,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        ).images

        for j, image in enumerate(images):
            prompt_id = id_list[j]
            image.save(os.path.join(output_path, f"{prompt_id:06d}.jpg"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption-path", type=str, default='../datasets/PartiPrompts/PartiPrompts.tsv')
    # 学长让先换到irip_16t_0
    parser.add_argument("--output-path", type=str, default="../../../../irip_16t_0/huangyu_2026/samples/PartiPrompts")
    parser.add_argument("--model-path", type=str, default="../checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--torch-dtype", type=str, default="float16")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tore-type", type=str, choices=["Default", "ToMe", "SDTM", "SDTM_TaylorSeer"], default="SDTM")
    
    # Additional ToMe arguments
    parser.add_argument("--ToMe-ratio", type=float, default=0.9)
    parser.add_argument("--ToMe-sx", type=int, default=2)
    parser.add_argument("--ToMe-sy", type=int, default=2)
    parser.add_argument("--ToMe-use-rand", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ToMe-merge-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ToMe-merge-mlp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ToMe-pseudo-merge", action=argparse.BooleanOptionalAction, default=True, help="Bind objects together without actual merging.")

    # Additional SDTM arguments
    parser.add_argument("--SDTM-ratio", type=float, default=0.3)
    parser.add_argument("--SDTM-deviation", type=float, default=0.2)
    parser.add_argument("--SDTM-switch-step", type=int, default=20)
    parser.add_argument("--SDTM-use-rand", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--SDTM-sx", type=int, default=2)
    parser.add_argument("--SDTM-sy", type=int, default=2)
    parser.add_argument("--SDTM-a-s", type=float, default=0.05)
    parser.add_argument("--SDTM-a-d", type=float, default=0.05)
    parser.add_argument("--SDTM-a-p", type=float, default=2)
    parser.add_argument("--SDTM-pseudo-merge", action=argparse.BooleanOptionalAction, default=False, help="Bind objects together without actual merging.")
    parser.add_argument("--SDTM-mcw", type=float, default=0.1, help="the weight for merge is w, while for cache is 1-w")
    parser.add_argument("--SDTM-protect-steps-frequency", type=int, default=3, help='Frequency for protecting steps')
    parser.add_argument("--SDTM-protect-layers-frequency", type=int, default=-1, help='Frequency for protecting layers')
    parser.add_argument("--SDTM-cache_each_step", action=argparse.BooleanOptionalAction, default=True, help="Bind objects together without actual merging.")

    # Additional TaylorSeer arguments (used with SDTM_TaylorSeer)
    parser.add_argument("--Taylor-interval", type=int, default=2, help="TaylorSeer caching interval: run full computation every N steps")
    parser.add_argument("--Taylor-max-order", type=int, default=1, help="Max order of Taylor expansion for derivative approximation")
    parser.add_argument("--Taylor-first-enhance", type=int, default=12, help="Number of initial/final steps forced to full computation")
    args = parser.parse_args()
    main(args)
