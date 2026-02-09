#!/usr/bin/env python3
"""
T2V-CompBench Video Generation Script

Generate videos for T2V-CompBench evaluation using Wan2.2 model.
Supports custom scheduler and default UniPC scheduler with configurable steps.
Supports multi-GPU parallel generation.

Output structure matches T2V-CompBench requirements:
    save_dir/
    ├── consistent_attr/
    │   ├── 0001.mp4
    │   ├── 0002.mp4
    │   └── ...
    ├── dynamic_attr/
    │   ├── 0001.mp4
    │   └── ...
    └── ...

Usage Examples:
    # Single GPU generation
    python t2v_compbench_generate.py \\
        --save_dir ./video/unipc_30steps \\
        --sampler unipc \\
        --num_steps 30

    # Multi-GPU generation (auto-detect all GPUs)
    python t2v_compbench_generate.py \\
        --save_dir ./video/custom_30steps \\
        --sampler custom \\
        --num_steps 30 \\
        --multi_gpu

    # Multi-GPU with specific GPU IDs
    python t2v_compbench_generate.py \\
        --save_dir ./video/custom_30steps \\
        --sampler custom \\
        --num_steps 30 \\
        --multi_gpu \\
        --gpu_ids 0 1 2 3
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import multiprocessing as mp
from functools import partial

import torch

from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video

# Import custom scheduler from stablevs
from stablevs import StableVSUniPCMultistepScheduler


# Category to prompt file mapping
CATEGORY_TO_FILE: Dict[str, str] = {
    "consistent_attr": "1_consistent_attr.txt",
    "dynamic_attr": "2_dynamic_attr.txt",
    "spatial_relationship": "3_spatial_relationship.txt",
    "motion_binding": "4_motion_binding.txt",
    "action_binding": "5_action_binding.txt",
    "interaction": "6_interaction.txt",
    "numeracy": "7_numeracy.txt",
}

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate videos for T2V-CompBench evaluation using Wan2.2 model."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save generated videos. Structure: save_dir/category/0001.mp4",
    )
    parser.add_argument(
        "--prompts_dir",
        type=str,
        default="./T2V-CompBench/prompts",
        help="Path to T2V-CompBench prompts directory.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=list(CATEGORY_TO_FILE.keys()),
        choices=list(CATEGORY_TO_FILE.keys()),
        help=(
            "Categories to generate videos for. Choices: "
            + ", ".join(CATEGORY_TO_FILE.keys())
        ),
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["unipc", "custom"],
        default="unipc",
        help="Sampler to use: 'unipc' (default) or 'custom'.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=30,
        help="Number of inference steps (default: 30).",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=1,
        help="Starting prompt index (1-based, default=1).",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="Ending prompt index (1-based, inclusive). If not set, process all prompts.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video frame height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Video frame width.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames per video.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="Frames per second for saved videos.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.0,
        help="Guidance scale for generation.",
    )
    parser.add_argument(
        "--guidance_scale_2",
        type=float,
        default=3.0,
        help="Second guidance scale for Wan2.2.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt to use during generation.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        help="Model ID for the Wan pipeline.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip generation if output file already exists.",
    )
    # Custom scheduler parameters
    parser.add_argument(
        "--fast_low_split_point",
        type=float,
        default=0.85,
        help="Split point for custom scheduler fast-low schedule.",
    )
    parser.add_argument(
        "--fast_low_low_substeps",
        type=int,
        default=9,
        help="Number of low substeps for custom scheduler.",
    )
    # Multi-GPU options
    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        help="Enable multi-GPU parallel generation.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=int,
        nargs="*",
        default=None,
        help="Specific GPU IDs to use. If not set, auto-detect all available GPUs.",
    )
    parser.add_argument(
        "--parallel_mode",
        type=str,
        choices=["category", "prompt"],
        default="prompt",
        help="Parallelization mode: 'category' (each GPU handles different categories) "
             "or 'prompt' (distribute prompts across GPUs within each category).",
    )
    return parser.parse_args()


def load_prompts_for_category(
    category: str, prompts_dir: str
) -> List[str]:
    """Load prompts from the category file."""
    if category not in CATEGORY_TO_FILE:
        raise ValueError(f"Unsupported category: {category}")
    
    filename = CATEGORY_TO_FILE[category]
    file_path = os.path.join(prompts_dir, filename)
    
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Prompt file not found for {category}: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.readlines()
    
    prompts = [line.strip() for line in raw if line.strip()]
    return prompts


def build_pipeline(
    model_id: str, device: str = "cuda:0"
) -> Tuple[WanPipeline, Any]:
    """Build the Wan pipeline and return it with the default scheduler."""
    torch_dtype = torch.bfloat16
    
    print(f"[{device}] Loading VAE from {model_id}...")
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32
    )
    
    print(f"[{device}] Loading pipeline from {model_id}...")
    pipe = WanPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=torch_dtype
    )
    pipe.to(device)
    
    default_scheduler = pipe.scheduler
    return pipe, default_scheduler


def build_custom_scheduler(args: argparse.Namespace) -> StableVSUniPCMultistepScheduler:
    """Build the custom scheduler with specified parameters."""
    return StableVSUniPCMultistepScheduler(
        beta_end=0.02,
        beta_schedule="linear",
        beta_start=0.0001,
        disable_corrector=[],
        dynamic_thresholding_ratio=0.995,
        final_sigmas_type="zero",
        flow_shift=3.0,
        lower_order_final=True,
        num_train_timesteps=1000,
        predict_x0=True,
        prediction_type="flow_prediction",
        rescale_betas_zero_snr=False,
        sample_max_value=1.0,
        solver_order=2,
        solver_p=None,
        solver_type="bh2",
        steps_offset=0,
        thresholding=False,
        time_shift_type="exponential",
        timestep_spacing="linspace",
        trained_betas=None,
        use_beta_sigmas=False,
        use_dynamic_shifting=False,
        use_exponential_sigmas=False,
        use_flow_sigmas=True,
        use_karras_sigmas=False,
        use_fast_low_schedule=True,
        fast_low_split_point=args.fast_low_split_point,
        fast_low_low_substeps=args.fast_low_low_substeps,
        low_region_noise_factor=0.0,
    )


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_video_filename(idx: int) -> str:
    """Generate video filename in T2V-CompBench format (0001.mp4, 0002.mp4, etc.)."""
    return f"{idx:04d}.mp4"


def get_available_gpus() -> List[int]:
    """Get list of available GPU IDs."""
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def worker_process_category(
    gpu_id: int,
    categories: List[str],
    args_dict: dict,
) -> dict:
    """Worker process for category-based parallelization."""
    device = f"cuda:{gpu_id}"
    results = {"generated": 0, "skipped": 0, "failed": 0}
    
    # Reconstruct args namespace
    args = argparse.Namespace(**args_dict)
    
    # Build pipeline on this GPU
    print(f"[GPU {gpu_id}] Initializing pipeline...")
    pipe, default_scheduler = build_pipeline(args.model_id, device=device)
    custom_scheduler = build_custom_scheduler(args)
    
    # Set scheduler
    if args.sampler == "custom":
        pipe.scheduler = custom_scheduler
    else:
        pipe.scheduler = default_scheduler
    
    # Prepare log file for this GPU
    log_csv_path = os.path.join(args.save_dir, f"generation_log_gpu{gpu_id}.csv")
    csv_file = open(log_csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "category", "prompt_idx", "prompt", "sampler", "num_steps",
        "seed", "height", "width", "num_frames", "fps", "filepath", "status",
    ])
    
    # Process assigned categories
    for category in categories:
        print(f"\n[GPU {gpu_id}] Processing category: {category}")
        
        try:
            prompts = load_prompts_for_category(category, args.prompts_dir)
        except Exception as e:
            print(f"[GPU {gpu_id}] ERROR: Failed to load prompts for '{category}': {e}")
            continue
        
        # Determine index range
        start_idx = args.start_idx
        end_idx = args.end_idx if args.end_idx is not None else len(prompts)
        end_idx = min(end_idx, len(prompts))
        
        # Create output directory
        out_dir = os.path.join(args.save_dir, category)
        ensure_dir(out_dir)
        
        # Process prompts
        for idx in range(start_idx, end_idx + 1):
            prompt = prompts[idx - 1]
            filename = get_video_filename(idx)
            out_path = os.path.join(out_dir, filename)
            
            # Skip if exists
            if args.skip_existing and os.path.exists(out_path):
                print(f"[GPU {gpu_id}] [{category}] [{idx}/{end_idx}] Skipping: {filename}")
                results["skipped"] += 1
                continue
            
            print(f"[GPU {gpu_id}] [{category}] [{idx}/{end_idx}] Generating: {filename}")
            
            try:
                generator = torch.Generator(device).manual_seed(args.seed)
                
                result = pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    guidance_scale=args.guidance_scale,
                    guidance_scale_2=args.guidance_scale_2,
                    num_inference_steps=args.num_steps,
                    generator=generator,
                )
                
                frames = result.frames[0]
                export_to_video(frames, out_path, fps=args.fps)
                
                csv_writer.writerow([
                    category, idx, prompt, args.sampler, args.num_steps,
                    args.seed, args.height, args.width, args.num_frames,
                    args.fps, out_path, "success",
                ])
                csv_file.flush()
                
                results["generated"] += 1
                print(f"[GPU {gpu_id}] ✓ Saved: {out_path}")
                
            except Exception as e:
                csv_writer.writerow([
                    category, idx, prompt, args.sampler, args.num_steps,
                    args.seed, args.height, args.width, args.num_frames,
                    args.fps, out_path, f"failed: {str(e)}",
                ])
                csv_file.flush()
                
                results["failed"] += 1
                print(f"[GPU {gpu_id}] ✗ Failed: {e}")
    
    csv_file.close()
    return results


def worker_process_prompts(
    gpu_id: int,
    category: str,
    prompt_indices: List[int],
    args_dict: dict,
) -> dict:
    """Worker process for prompt-based parallelization."""
    device = f"cuda:{gpu_id}"
    results = {"generated": 0, "skipped": 0, "failed": 0}
    
    # Reconstruct args namespace
    args = argparse.Namespace(**args_dict)
    
    # Build pipeline on this GPU
    print(f"[GPU {gpu_id}] Initializing pipeline for {category}...")
    pipe, default_scheduler = build_pipeline(args.model_id, device=device)
    custom_scheduler = build_custom_scheduler(args)
    
    # Set scheduler
    if args.sampler == "custom":
        pipe.scheduler = custom_scheduler
    else:
        pipe.scheduler = default_scheduler
    
    # Load prompts
    prompts = load_prompts_for_category(category, args.prompts_dir)
    
    # Create output directory
    out_dir = os.path.join(args.save_dir, category)
    ensure_dir(out_dir)
    
    # Prepare log file
    log_csv_path = os.path.join(args.save_dir, f"generation_log_{category}_gpu{gpu_id}.csv")
    csv_file = open(log_csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "category", "prompt_idx", "prompt", "sampler", "num_steps",
        "seed", "height", "width", "num_frames", "fps", "filepath", "status",
    ])
    
    total = len(prompt_indices)
    for i, idx in enumerate(prompt_indices):
        prompt = prompts[idx - 1]
        filename = get_video_filename(idx)
        out_path = os.path.join(out_dir, filename)
        
        # Skip if exists
        if args.skip_existing and os.path.exists(out_path):
            print(f"[GPU {gpu_id}] [{category}] [{i+1}/{total}] Skipping: {filename}")
            results["skipped"] += 1
            continue
        
        print(f"[GPU {gpu_id}] [{category}] [{i+1}/{total}] Generating: {filename} (prompt {idx})")
        
        try:
            generator = torch.Generator(device).manual_seed(args.seed)
            
            result = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                guidance_scale=args.guidance_scale,
                guidance_scale_2=args.guidance_scale_2,
                num_inference_steps=args.num_steps,
                generator=generator,
            )
            
            frames = result.frames[0]
            export_to_video(frames, out_path, fps=args.fps)
            
            csv_writer.writerow([
                category, idx, prompt, args.sampler, args.num_steps,
                args.seed, args.height, args.width, args.num_frames,
                args.fps, out_path, "success",
            ])
            csv_file.flush()
            
            results["generated"] += 1
            print(f"[GPU {gpu_id}] ✓ Saved: {out_path}")
            
        except Exception as e:
            csv_writer.writerow([
                category, idx, prompt, args.sampler, args.num_steps,
                args.seed, args.height, args.width, args.num_frames,
                args.fps, out_path, f"failed: {str(e)}",
            ])
            csv_file.flush()
            
            results["failed"] += 1
            print(f"[GPU {gpu_id}] ✗ Failed: {e}")
    
    csv_file.close()
    return results


def distribute_categories(categories: List[str], num_gpus: int) -> List[List[str]]:
    """Distribute categories across GPUs as evenly as possible."""
    distribution = [[] for _ in range(num_gpus)]
    for i, category in enumerate(categories):
        distribution[i % num_gpus].append(category)
    return distribution


def distribute_prompts(start_idx: int, end_idx: int, num_gpus: int) -> List[List[int]]:
    """Distribute prompt indices across GPUs as evenly as possible."""
    all_indices = list(range(start_idx, end_idx + 1))
    distribution = [[] for _ in range(num_gpus)]
    for i, idx in enumerate(all_indices):
        distribution[i % num_gpus].append(idx)
    return distribution


def run_multi_gpu_category_mode(args: argparse.Namespace, gpu_ids: List[int]) -> None:
    """Run multi-GPU generation with category-based parallelization."""
    num_gpus = len(gpu_ids)
    category_distribution = distribute_categories(args.categories, num_gpus)
    
    print(f"\nCategory distribution across {num_gpus} GPUs:")
    for i, (gpu_id, cats) in enumerate(zip(gpu_ids, category_distribution)):
        print(f"  GPU {gpu_id}: {', '.join(cats) if cats else '(none)'}")
    
    # Convert args to dict for multiprocessing
    args_dict = vars(args)
    
    # Start worker processes
    mp.set_start_method('spawn', force=True)
    
    with mp.Pool(processes=num_gpus) as pool:
        worker_args = [
            (gpu_id, cats, args_dict)
            for gpu_id, cats in zip(gpu_ids, category_distribution)
            if cats  # Only spawn workers for GPUs with work
        ]
        
        results = pool.starmap(worker_process_category, worker_args)
    
    # Aggregate results
    total_generated = sum(r["generated"] for r in results)
    total_skipped = sum(r["skipped"] for r in results)
    total_failed = sum(r["failed"] for r in results)
    
    print(f"\n{'='*60}")
    print("Multi-GPU Generation Summary (Category Mode)")
    print(f"{'='*60}")
    print(f"  GPUs used:       {num_gpus}")
    print(f"  Sampler:         {args.sampler}")
    print(f"  Num steps:       {args.num_steps}")
    print(f"  Total generated: {total_generated}")
    print(f"  Total skipped:   {total_skipped}")
    print(f"  Total failed:    {total_failed}")
    print(f"{'='*60}")


def run_multi_gpu_prompt_mode(args: argparse.Namespace, gpu_ids: List[int]) -> None:
    """Run multi-GPU generation with prompt-based parallelization."""
    num_gpus = len(gpu_ids)
    args_dict = vars(args)
    
    mp.set_start_method('spawn', force=True)
    
    total_generated = 0
    total_skipped = 0
    total_failed = 0
    
    for category in args.categories:
        print(f"\n{'='*60}")
        print(f"Processing category: {category} (across {num_gpus} GPUs)")
        print(f"{'='*60}")
        
        # Load prompts to get count
        prompts = load_prompts_for_category(category, args.prompts_dir)
        start_idx = args.start_idx
        end_idx = args.end_idx if args.end_idx is not None else len(prompts)
        end_idx = min(end_idx, len(prompts))
        
        # Distribute prompts
        prompt_distribution = distribute_prompts(start_idx, end_idx, num_gpus)
        
        print(f"Prompt distribution:")
        for gpu_id, indices in zip(gpu_ids, prompt_distribution):
            if indices:
                print(f"  GPU {gpu_id}: prompts {indices[0]}-{indices[-1]} ({len(indices)} total)")
        
        # Run workers
        with mp.Pool(processes=num_gpus) as pool:
            worker_args = [
                (gpu_id, category, indices, args_dict)
                for gpu_id, indices in zip(gpu_ids, prompt_distribution)
                if indices
            ]
            
            results = pool.starmap(worker_process_prompts, worker_args)
        
        # Aggregate results for this category
        cat_generated = sum(r["generated"] for r in results)
        cat_skipped = sum(r["skipped"] for r in results)
        cat_failed = sum(r["failed"] for r in results)
        
        print(f"Category '{category}' complete: generated={cat_generated}, skipped={cat_skipped}, failed={cat_failed}")
        
        total_generated += cat_generated
        total_skipped += cat_skipped
        total_failed += cat_failed
    
    print(f"\n{'='*60}")
    print("Multi-GPU Generation Summary (Prompt Mode)")
    print(f"{'='*60}")
    print(f"  GPUs used:       {num_gpus}")
    print(f"  Sampler:         {args.sampler}")
    print(f"  Num steps:       {args.num_steps}")
    print(f"  Total generated: {total_generated}")
    print(f"  Total skipped:   {total_skipped}")
    print(f"  Total failed:    {total_failed}")
    print(f"{'='*60}")


def run_single_gpu(args: argparse.Namespace) -> None:
    """Run single GPU generation."""
    save_root = os.path.abspath(args.save_dir)
    ensure_dir(save_root)
    
    # Build pipeline
    print("Initializing pipeline...")
    pipe, default_scheduler = build_pipeline(args.model_id, device="cuda:0")
    custom_scheduler = build_custom_scheduler(args)
    
    # Set scheduler
    if args.sampler == "custom":
        pipe.scheduler = custom_scheduler
        print("Using: Custom scheduler")
    else:
        pipe.scheduler = default_scheduler
        print("Using: UniPC scheduler (default)")
    
    # Prepare CSV for logging
    log_csv_path = os.path.join(save_root, "generation_log.csv")
    csv_file = open(log_csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    
    if os.path.getsize(log_csv_path) == 0:
        csv_writer.writerow([
            "category", "prompt_idx", "prompt", "sampler", "num_steps",
            "seed", "height", "width", "num_frames", "fps", "filepath", "status",
        ])
    
    total_generated = 0
    total_skipped = 0
    total_failed = 0
    
    for category in args.categories:
        print(f"\n{'='*60}")
        print(f"Processing category: {category}")
        print(f"{'='*60}")
        
        try:
            prompts = load_prompts_for_category(category, args.prompts_dir)
        except Exception as e:
            print(f"[ERROR] Failed to load prompts for '{category}': {e}")
            continue
        
        start_idx = args.start_idx
        end_idx = args.end_idx if args.end_idx is not None else len(prompts)
        end_idx = min(end_idx, len(prompts))
        
        print(f"Processing prompts {start_idx} to {end_idx}")
        
        out_dir = os.path.join(save_root, category)
        ensure_dir(out_dir)
        
        for idx in range(start_idx, end_idx + 1):
            prompt = prompts[idx - 1]
            filename = get_video_filename(idx)
            out_path = os.path.join(out_dir, filename)
            
            if args.skip_existing and os.path.exists(out_path):
                print(f"  [{idx}/{end_idx}] Skipping: {filename}")
                total_skipped += 1
                continue
            
            print(f"  [{idx}/{end_idx}] Generating: {filename}")
            print(f"    Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
            
            try:
                generator = torch.Generator("cuda:0").manual_seed(args.seed)
                
                result = pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    guidance_scale=args.guidance_scale,
                    guidance_scale_2=args.guidance_scale_2,
                    num_inference_steps=args.num_steps,
                    generator=generator,
                )
                
                frames = result.frames[0]
                export_to_video(frames, out_path, fps=args.fps)
                
                csv_writer.writerow([
                    category, idx, prompt, args.sampler, args.num_steps,
                    args.seed, args.height, args.width, args.num_frames,
                    args.fps, out_path, "success",
                ])
                csv_file.flush()
                
                total_generated += 1
                print(f"    ✓ Saved: {out_path}")
                
            except Exception as e:
                csv_writer.writerow([
                    category, idx, prompt, args.sampler, args.num_steps,
                    args.seed, args.height, args.width, args.num_frames,
                    args.fps, out_path, f"failed: {str(e)}",
                ])
                csv_file.flush()
                
                total_failed += 1
                print(f"    ✗ Failed: {e}")
    
    csv_file.close()
    
    print(f"\n{'='*60}")
    print("Generation Summary")
    print(f"{'='*60}")
    print(f"  Sampler:         {args.sampler}")
    print(f"  Num steps:       {args.num_steps}")
    print(f"  Total generated: {total_generated}")
    print(f"  Total skipped:   {total_skipped}")
    print(f"  Total failed:    {total_failed}")
    print(f"  Log file:        {log_csv_path}")
    print(f"{'='*60}")


def main() -> None:
    args = parse_args()
    
    save_root = os.path.abspath(args.save_dir)
    ensure_dir(save_root)
    
    # Print configuration
    print(f"\n{'='*60}")
    print("T2V-CompBench Video Generation")
    print(f"{'='*60}")
    print(f"  Sampler:     {args.sampler}")
    print(f"  Num steps:   {args.num_steps}")
    print(f"  Categories:  {', '.join(args.categories)}")
    print(f"  Output dir:  {save_root}")
    print(f"  Resolution:  {args.width}x{args.height}")
    print(f"  Num frames:  {args.num_frames}")
    print(f"  FPS:         {args.fps}")
    print(f"  Seed:        {args.seed}")
    print(f"  Multi-GPU:   {args.multi_gpu}")
    
    if args.multi_gpu:
        # Determine GPU IDs
        if args.gpu_ids is not None:
            gpu_ids = args.gpu_ids
        else:
            gpu_ids = get_available_gpus()
        
        if not gpu_ids:
            print("ERROR: No GPUs available!")
            sys.exit(1)
        
        print(f"  GPU IDs:     {gpu_ids}")
        print(f"  Parallel:    {args.parallel_mode} mode")
        print(f"{'='*60}\n")
        
        if args.parallel_mode == "category":
            run_multi_gpu_category_mode(args, gpu_ids)
        else:
            run_multi_gpu_prompt_mode(args, gpu_ids)
    else:
        print(f"{'='*60}\n")
        run_single_gpu(args)


if __name__ == "__main__":
    main()
