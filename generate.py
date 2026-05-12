import json
import argparse
import torch
import os
import random

from tqdm import tqdm
from PIL import Image, ImageDraw

from TR_ToMe import apply_ToMe
from TR_SDTM import apply_SDTM
from TR_SDTM_TaylorSeer import apply_SDTM_TaylorSeer

from diffusers import StableDiffusion3Pipeline


def decode_latents_to_pil(pipe, latents):
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    with torch.no_grad():
        image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    return image


def make_blocky_preview(
    image,
    layout="regular",
    grid_size=3,
    gap=6,
    padding=6,
    background=(255, 255, 255),
    subdivide_index=4,
    subdivide_grid=2,
    subdivide_gap=4,
    irregular_max_depth=3,
    irregular_min_tile=128,
    irregular_split_prob=0.9,
    irregular_seed=0,
):
    if layout == "irregular":
        return make_irregular_preview(
            image=image,
            gap=gap,
            padding=padding,
            background=background,
            max_depth=irregular_max_depth,
            min_tile=irregular_min_tile,
            split_prob=irregular_split_prob,
            seed=irregular_seed,
        )

    if grid_size <= 1:
        return image

    width, height = image.size
    cell_width = width // grid_size
    cell_height = height // grid_size

    canvas_width = padding * 2 + grid_size * cell_width + (grid_size - 1) * gap
    canvas_height = padding * 2 + grid_size * cell_height + (grid_size - 1) * gap
    canvas = Image.new("RGB", (canvas_width, canvas_height), background)

    for row in range(grid_size):
        for col in range(grid_size):
            left = col * cell_width
            upper = row * cell_height
            right = width if col == grid_size - 1 else (col + 1) * cell_width
            lower = height if row == grid_size - 1 else (row + 1) * cell_height

            tile = image.crop((left, upper, right, lower))
            tile_x = padding + col * (cell_width + gap)
            tile_y = padding + row * (cell_height + gap)

            tile_index = row * grid_size + col
            if tile_index == subdivide_index and subdivide_grid > 1:
                inner_gap = max(0, subdivide_gap)
                sub_w = (cell_width - inner_gap * (subdivide_grid - 1)) // subdivide_grid
                sub_h = (cell_height - inner_gap * (subdivide_grid - 1)) // subdivide_grid
                sub_w = max(1, sub_w)
                sub_h = max(1, sub_h)

                for sr in range(subdivide_grid):
                    for sc in range(subdivide_grid):
                        src_left = int(sc * tile.width / subdivide_grid)
                        src_right = int((sc + 1) * tile.width / subdivide_grid)
                        src_upper = int(sr * tile.height / subdivide_grid)
                        src_lower = int((sr + 1) * tile.height / subdivide_grid)

                        sub_tile = tile.crop((src_left, src_upper, src_right, src_lower))
                        sub_tile = sub_tile.resize((sub_w, sub_h), resample=Image.LANCZOS)

                        dst_x = tile_x + sc * (sub_w + inner_gap)
                        dst_y = tile_y + sr * (sub_h + inner_gap)
                        canvas.paste(sub_tile, (dst_x, dst_y))
            else:
                tile = tile.resize((cell_width, cell_height), resample=Image.LANCZOS)
                canvas.paste(tile, (tile_x, tile_y))

    return canvas


def make_irregular_preview(
    image,
    gap=6,
    padding=6,
    background=(255, 255, 255),
    max_depth=3,
    min_tile=128,
    split_prob=0.9,
    seed=0,
):
    width, height = image.size
    rng = random.Random(seed)

    stack = [(0, 0, width, height, 0)]
    leaves = []

    while stack:
        x0, y0, x1, y1, depth = stack.pop()
        w = x1 - x0
        h = y1 - y0

        can_split = (
            depth < max_depth
            and w >= 2 * min_tile
            and h >= 2 * min_tile
            and rng.random() < split_prob
        )

        if not can_split:
            leaves.append((x0, y0, x1, y1))
            continue

        if w / max(h, 1) > 1.2:
            split_vertical = True
        elif h / max(w, 1) > 1.2:
            split_vertical = False
        else:
            split_vertical = rng.random() < 0.5

        if split_vertical:
            cut_min = x0 + min_tile
            cut_max = x1 - min_tile
            if cut_min >= cut_max:
                leaves.append((x0, y0, x1, y1))
                continue
            cut = rng.randint(cut_min, cut_max)
            stack.append((x0, y0, cut, y1, depth + 1))
            stack.append((cut, y0, x1, y1, depth + 1))
        else:
            cut_min = y0 + min_tile
            cut_max = y1 - min_tile
            if cut_min >= cut_max:
                leaves.append((x0, y0, x1, y1))
                continue
            cut = rng.randint(cut_min, cut_max)
            stack.append((x0, y0, x1, cut, depth + 1))
            stack.append((x0, cut, x1, y1, depth + 1))

    canvas = Image.new("RGB", (width + 2 * padding, height + 2 * padding), background)

    inset = max(0, gap // 2)
    for x0, y0, x1, y1 in leaves:
        tile = image.crop((x0, y0, x1, y1))
        dst_w = max(1, (x1 - x0) - gap)
        dst_h = max(1, (y1 - y0) - gap)
        tile = tile.resize((dst_w, dst_h), resample=Image.LANCZOS)
        dx = padding + x0 + inset
        dy = padding + y0 + inset
        canvas.paste(tile, (dx, dy))

    return canvas


def resize_contain(image, target_size):
    target_width, target_height = target_size
    ratio = min(target_width / image.width, target_height / image.height)
    new_size = (
        max(1, int(round(image.width * ratio))),
        max(1, int(round(image.height * ratio))),
    )
    return image.resize(new_size, resample=Image.LANCZOS)


def build_stage_montage(stage_images, output_path, grid_size=3, inner_grid_size=2, gap=12, padding=12):
    if not stage_images:
        return None

    total_panels = grid_size * grid_size
    selected = []
    if len(stage_images) >= total_panels:
        step = (len(stage_images) - 1) / max(total_panels - 1, 1)
        for idx in range(total_panels):
            selected.append(stage_images[int(round(idx * step))])
    else:
        selected = list(stage_images)
        while len(selected) < total_panels:
            selected.append(selected[-1])

    sample_width, sample_height = selected[0].size
    panel_width = sample_width * inner_grid_size + gap * (inner_grid_size - 1)
    panel_height = sample_height * inner_grid_size + gap * (inner_grid_size - 1)

    canvas_width = padding * 2 + grid_size * panel_width + (grid_size - 1) * gap
    canvas_height = padding * 2 + grid_size * panel_height + (grid_size - 1) * gap
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))

    for panel_index, stage_image in enumerate(selected):
        row = panel_index // grid_size
        col = panel_index % grid_size

        panel_x = padding + col * (panel_width + gap)
        panel_y = padding + row * (panel_height + gap)

        base_stage = stage_image.copy()
        stage_variants = [
            make_blocky_preview(base_stage, grid_size=3, gap=max(4, gap // 2), padding=max(4, padding // 2)),
            make_blocky_preview(base_stage, grid_size=3, gap=max(4, gap // 2), padding=max(4, padding // 2)),
            base_stage,
            base_stage,
        ]

        for inner_index, variant in enumerate(stage_variants[: inner_grid_size * inner_grid_size]):
            inner_row = inner_index // inner_grid_size
            inner_col = inner_index % inner_grid_size
            cell_x = panel_x + inner_col * (sample_width + gap)
            cell_y = panel_y + inner_row * (sample_height + gap)

            fitted = resize_contain(variant, (sample_width, sample_height))
            paste_x = cell_x + (sample_width - fitted.width) // 2
            paste_y = cell_y + (sample_height - fitted.height) // 2
            canvas.paste(fitted, (paste_x, paste_y))

    montage_path = os.path.join(output_path, "stages_montage.jpg")
    canvas.save(montage_path)
    return montage_path

def load_captions(file_path):
    """
    Reads the 'annotations' section of a specified JSON file, extracts the image_id and caption from each annotation,
    and returns a list containing this data, ensuring that each image_id is unique and has the longest caption.

    :param file_path: Path to the JSON file
    :return: A list where each element is a dictionary containing 'image_id' and the longest 'caption' for that image_id
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    annotations = data
    # Dictionary to store the longest caption for each image_id
    longest_captions = {}

    # Iterate through each annotation
    for item in annotations:
        image_id = item['image_id']
        caption = item['caption']
        # If the image_id is already in the dictionary and the current caption is longer, update it
        if image_id in longest_captions:
            if len(caption) > len(longest_captions[image_id]['caption']):
                longest_captions[image_id] = {'image_id': image_id, 'caption': caption}
        else:
            # Otherwise, add the image_id and caption to the dictionary
            longest_captions[image_id] = {'image_id': image_id, 'caption': caption}

    # Extract values from the dictionary to form the final list
    image_captions = list(longest_captions.values())

    return image_captions

def main(args):
    """
    Run sampling on single GPU.
    """
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    file_path = args.caption_path
    captions_list = load_captions(file_path)
    
    # 保存为 JSON 文件
    with open(args.caption_path, 'w', encoding='utf-8') as file:
        json.dump(captions_list, file, ensure_ascii=False, indent=4)

    print(f"Loaded unduplicated {len(captions_list)} captions.")

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

    single_prompt = getattr(args, "caption", "").strip()
    if single_prompt:
        print(f"Prompt: {single_prompt}")

        stage_dir = os.path.join(output_path, "stages")
        if args.save_stages:
            os.makedirs(stage_dir, exist_ok=True)

        saved_steps = set()
        stage_images = []

        def save_stage(step_index, latents):
            if not args.save_stages:
                return

            if step_index in saved_steps:
                return

            should_save = (
                step_index < args.stage_first_steps
                or step_index % max(args.stage_interval, 1) == 0
                or step_index == args.num_inference_steps - 1
            )
            if not should_save:
                return

            stage_image = decode_latents_to_pil(pipe, latents)
            stage_image = make_blocky_preview(
                stage_image,
                layout=args.stage_layout,
                grid_size=args.stage_grid_size,
                gap=args.stage_gap,
                padding=args.stage_padding,
                subdivide_index=args.stage_subdivide_index,
                subdivide_grid=args.stage_subdivide_grid,
                subdivide_gap=args.stage_subdivide_gap,
                irregular_max_depth=args.stage_irregular_max_depth,
                irregular_min_tile=args.stage_irregular_min_tile,
                irregular_split_prob=args.stage_irregular_split_prob,
                irregular_seed=args.seed * 10000 + step_index,
            )
            stage_path = os.path.join(stage_dir, f"step_{step_index:03d}.jpg")
            stage_image.save(stage_path)
            stage_images.append(stage_image)
            saved_steps.add(step_index)

        def callback_on_step_end(pipe_obj, step_index, timestep, callback_kwargs):
            save_stage(step_index, callback_kwargs["latents"])
            return callback_kwargs

        with torch.no_grad():
            image = pipe(
                prompt=single_prompt,
                generator=generator,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
            ).images[0]

        image.save(os.path.join(output_path, "final.jpg"))
        montage_path = build_stage_montage(
            stage_images,
            output_path=output_path,
            grid_size=args.montage_grid_size,
            inner_grid_size=args.montage_inner_grid_size,
            gap=args.montage_gap,
            padding=args.montage_padding,
        )
        print(f"Saved final image and stage previews to: {output_path}")
        if montage_path:
            print(f"Saved montage: {montage_path}")
        return

    # Warmup
    print("\n---Warming up the model---")
    warmup_prompt = "First-person view from inside a self-driving car, cruising on a highway at sunset, no driver controlling the vehicle, hands resting on lap, dashboard autonomy display, modern interior, realistic road scene"
    with torch.no_grad():
        images = pipe(
            prompt=warmup_prompt,
            generator=generator,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        ).images
    print("---Warmup Completed---\n")

    for j, image in enumerate(images):
        image_id = "4"
        image.save(os.path.join(output_path, f"{image_id}.jpg"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption", type=str, default="", help="Single prompt to generate; if empty, use caption_path batch mode.")
    # parser.add_argument("--caption-path", type=str, default='datasets/COCO2017/longest_captions.json')
    parser.add_argument("--caption-path", type=str, default='datasets/COCO2017/captions_val2017.json')
    # 学长让先换到irip_16t_0
    parser.add_argument("--output-path", type=str, default="../../../irip_16t_0/huangyu_2026/samples")
    parser.add_argument("--model-path", type=str, default="checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--torch-dtype", type=str, default="float16")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tore-type", type=str, choices=["Default", "ToMe", "SDTM", "SDTM_TaylorSeer"], default="SDTM")
    parser.add_argument("--save-stages", action=argparse.BooleanOptionalAction, default=True, help="Save intermediate stage previews in single-prompt mode.")
    parser.add_argument("--stage-interval", type=int, default=4, help="Save one stage preview every N denoising steps.")
    parser.add_argument("--stage-first-steps", type=int, default=3, help="Always save the first N denoising steps.")
    parser.add_argument("--stage-layout", type=str, choices=["regular", "irregular"], default="irregular", help="Stage block layout style.")
    parser.add_argument("--stage-grid-size", type=int, default=3, help="Grid size for stage previews; 3 means split into 9 big blocks.")
    parser.add_argument("--stage-gap", type=int, default=6, help="Gap between the 3x3 blocks.")
    parser.add_argument("--stage-padding", type=int, default=6, help="Padding around the 3x3 preview canvas.")
    parser.add_argument("--stage-subdivide-index", type=int, default=4, help="Which big block to subdivide (0-based in row-major order). 4 means center block.")
    parser.add_argument("--stage-subdivide-grid", type=int, default=2, help="Subdivide selected big block into NxN; 2 means 4 small blocks.")
    parser.add_argument("--stage-subdivide-gap", type=int, default=4, help="Gap between subdivided small blocks.")
    parser.add_argument("--stage-irregular-max-depth", type=int, default=3, help="Max recursive split depth for irregular layout.")
    parser.add_argument("--stage-irregular-min-tile", type=int, default=128, help="Minimum tile size for irregular layout.")
    parser.add_argument("--stage-irregular-split-prob", type=float, default=0.9, help="Split probability for irregular layout.")
    parser.add_argument("--montage-grid-size", type=int, default=3, help="Outer grid size for the montage (kept for compatibility).")
    parser.add_argument("--montage-inner-grid-size", type=int, default=2, help="Inner grid size for each montage block (kept for compatibility).")
    parser.add_argument("--montage-gap", type=int, default=12, help="Gap between blocks in the montage.")
    parser.add_argument("--montage-padding", type=int, default=12, help="Padding around the montage canvas.")
    
    # Additional ToMe arguments
    parser.add_argument("--ToMe-ratio", type=float, default=0.1)
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
    parser.add_argument("--Taylor-interval", type=int, default=4, help="TaylorSeer caching interval: run full computation every N steps")
    parser.add_argument("--Taylor-max-order", type=int, default=2, help="Max order of Taylor expansion for derivative approximation")
    parser.add_argument("--Taylor-first-enhance", type=int, default=2, help="Number of initial/final steps forced to full computation")
    args = parser.parse_args()
    main(args)
