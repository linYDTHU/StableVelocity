import argparse
import os
from multiprocessing import Process, set_start_method
from typing import Dict, List, Optional, Tuple

import torch
from diffusers import DiffusionPipeline, FluxPipeline, StableDiffusion3Pipeline
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

# Import custom schedulers from stablevs
from stablevs import StableVSFlowMatchScheduler, StableVSDPMSolverMultistepScheduler


def ensure_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_prompts(file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> List[Tuple[int, str]]:
    with open(file_path, "r", encoding="utf-8") as f:
        all_lines = [line.rstrip("\n") for line in f.readlines()]

    total_lines = len(all_lines)
    first_index = 1 if start_line is None else max(1, start_line)
    last_index = total_lines if end_line is None else min(total_lines, end_line)

    # Return pairs of (original 1-based line number, prompt)
    return [(i, all_lines[i - 1]) for i in range(first_index, last_index + 1)]


def create_custom_scheduler_for_model(
    model_key: str,
    device: str,
    fast_low_split_point: float,
    fast_low_low_substeps: int,
    low_region_noise_factor: float,
) -> StableVSFlowMatchScheduler:
    if model_key == "flux":
        return StableVSFlowMatchScheduler(
            base_image_seq_len=256,
            max_image_seq_len=4096,
            use_dynamic_shifting=True,
            base_shift=0.5,
            max_shift=1.15,
            shift=3.0,
            num_train_timesteps=1000,
            use_fast_low_schedule=True,
            fast_low_split_point=fast_low_split_point,
            fast_low_low_substeps=fast_low_low_substeps,
            low_region_noise_factor=low_region_noise_factor,
        )
    elif model_key == "qwen":
        # Qwen Image 2512 uses exponential time shift with specific parameters
        return StableVSFlowMatchScheduler(
            base_image_seq_len=256,
            max_image_seq_len=8192,
            use_dynamic_shifting=True,
            base_shift=0.5,
            max_shift=0.9,
            shift=1.0,
            shift_terminal=0.02,
            time_shift_type="exponential",
            num_train_timesteps=1000,
            use_fast_low_schedule=True,
            fast_low_split_point=fast_low_split_point,
            fast_low_low_substeps=fast_low_low_substeps,
            low_region_noise_factor=low_region_noise_factor,
        )
    else:
        # sd3 and sd35 use a simpler custom configuration as per example
        return StableVSFlowMatchScheduler(
            shift=3.0,
            num_train_timesteps=1000,
            use_fast_low_schedule=True,
            fast_low_split_point=fast_low_split_point,
            fast_low_low_substeps=fast_low_low_substeps,
            low_region_noise_factor=low_region_noise_factor,
        )


def create_dpm_solver_scheduler(
    sampler: str,
    fast_low_split_point: float,
    fast_low_low_substeps: int,
    low_region_noise_factor: float,
) -> torch.nn.Module:
    """
    Build a DPM solver scheduler (original or custom) using the provided configuration in SANA (https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers/tree/main/scheduler).
    Default configuration is set to the provided JSON in the request.
    """
    dpm_cfg = dict(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        solver_order=2,
        prediction_type="flow_prediction",
        thresholding=False,
        dynamic_thresholding_ratio=0.995,
        sample_max_value=1.0,
        algorithm_type="dpmsolver++",
        solver_type="midpoint",
        lower_order_final=True,
        euler_at_final=False,
        use_karras_sigmas=False,
        use_exponential_sigmas=False,
        use_beta_sigmas=False,
        use_lu_lambdas=False,
        use_flow_sigmas=True,
        flow_shift=3.0,
        final_sigmas_type="zero",
        lambda_min_clipped=-float("inf"),
        variance_type=None,
        timestep_spacing="linspace",
        steps_offset=0,
        rescale_betas_zero_snr=False,
    )

    if sampler == "custom_dpm":
        # enable fast-low schedule parameters for the custom version
        return StableVSDPMSolverMultistepScheduler(
            **dpm_cfg,
            use_fast_low_schedule=True,
            fast_low_split_point=fast_low_split_point,
            fast_low_low_substeps=fast_low_low_substeps,
            low_region_noise_factor=low_region_noise_factor,
        )
    else:
        # original DPM solver without fast-low customization
        return DPMSolverMultistepScheduler(**dpm_cfg)


def build_model_configs() -> Dict[str, Dict]:
    return {
        "flux": {
            "pipeline_cls": FluxPipeline,
            "model_id": "black-forest-labs/FLUX.1-dev",
            "dtype": torch.bfloat16,
            "guidance_scale": 3.5,
        },
        "sd3": {
            "pipeline_cls": StableDiffusion3Pipeline,
            "model_id": "stabilityai/stable-diffusion-3-medium-diffusers",
            "dtype": torch.float16,
            "guidance_scale": 5.0,
        },
        "sd35": {
            "pipeline_cls": StableDiffusion3Pipeline,
            "model_id": "stabilityai/stable-diffusion-3.5-large",
            "dtype": torch.bfloat16,
            "guidance_scale": 4.5,
        },
        "qwen": {
            "pipeline_cls": DiffusionPipeline,
            "model_id": "Qwen/Qwen-Image-2512",
            "dtype": torch.bfloat16,
            "guidance_scale": 4.0,
            "use_true_cfg": True,
            "prompt_suffix": "",
            "negative_prompt": "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch image generation for Flux, SD3, and SD3.5 with multiple configs.")
    parser.add_argument("--prompts", type=str, default="generation_prompts.txt", help="Path to prompts file")
    parser.add_argument("--output-dir", type=str, default="samples", help="Output directory root")
    parser.add_argument(
        "--models",
        type=str,
        default="flux,sd35,qwen",
        help="Comma-separated list of models to run: flux,sd3,sd35,qwen",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=30,
        help="Number of inference steps (e.g., 30).",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="euler",
        choices=["euler", "custom_euler", "dpm", "custom_dpm"],
        help="Sampler to use: euler (default), custom_euler (CustomFlowMatchScheduler), dpm (DPMSolver), custom_dpm (CustomDPMSolver).",
    )
    parser.add_argument("--start-line", type=int, default=None, help="1-based start line (inclusive)")
    parser.add_argument("--end-line", type=int, default=None, help="1-based end line (inclusive)")
    parser.add_argument("--skip-empty", action="store_true", help="Skip empty prompt lines")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or cpu)")
    parser.add_argument("--dry-run", action="store_true", help="Only create folders and print plan, do not generate")
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs (e.g., '0,1,2,3'). If not set, auto-detects all available GPUs.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable auto-resume (regenerate even if output file exists).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[512, 1024],
        default=1024,
        help="Square output resolution. Choose 1024 (default) or 512.",
    )
    parser.add_argument(
        "--fast-low-split-point",
        dest="fast_low_split_point",
        type=float,
        default=0.85,
        help="Custom scheduler: split point for fast-low schedule (default: 0.85)",
    )
    parser.add_argument(
        "--fast-low-low-substeps",
        dest="fast_low_low_substeps",
        type=int,
        default=9,
        help="Custom scheduler: number of low region substeps (default: 8)",
    )
    parser.add_argument(
        "--low-region-noise-factor",
        dest="low_region_noise_factor",
        type=float,
        default=0.0,
        help="Custom scheduler: noise factor for low region (default: 0.2)",
    )
    return parser.parse_args()


def _fmt_float_tag(value: float) -> str:
    """Format float for directory naming."""
    s = f"{value:.2f}"
    s = s.rstrip("0").rstrip(".")
    return s


def get_output_dir(args: argparse.Namespace, model_key: str) -> str:
    """Compute output directory path based on args and model."""
    num_inference_steps = int(args.num_steps)
    sampler_tag = args.sampler
    steps_tag = f"{num_inference_steps}steps"
    if args.sampler in ("custom_euler", "custom_dpm"):
        cfg_suffix = f"{steps_tag}-{sampler_tag}-sp{_fmt_float_tag(args.fast_low_split_point)}-sb{args.fast_low_low_substeps}-nf{_fmt_float_tag(args.low_region_noise_factor)}"
        return os.path.join(args.output_dir, model_key, cfg_suffix)
    else:
        return os.path.join(args.output_dir, model_key, f"{steps_tag}-{sampler_tag}")


def worker(
    gpu_id: int,
    args: argparse.Namespace,
    model_key: str,
    prompts_subset: List[Tuple[int, str]],
    seeds: List[int],
) -> None:
    """Worker function that runs on a single GPU."""
    device = f"cuda:{gpu_id}"
    model_registry = build_model_configs()
    model_cfg = model_registry[model_key]
    pipeline_cls = model_cfg["pipeline_cls"]
    model_id = model_cfg["model_id"]
    dtype = model_cfg["dtype"]
    guidance_scale = model_cfg["guidance_scale"]
    use_true_cfg = model_cfg.get("use_true_cfg", False)
    prompt_suffix = model_cfg.get("prompt_suffix", "")
    negative_prompt = model_cfg.get("negative_prompt", None)

    height = args.resolution
    width = args.resolution
    num_inference_steps = int(args.num_steps)
    auto_resume = not args.no_resume

    out_dir = get_output_dir(args, model_key)
    ensure_directory(out_dir)

    # Load pipeline
    pipe = pipeline_cls.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    default_scheduler = pipe.scheduler

    # Choose scheduler based on sampler
    if args.sampler == "euler":
        pipe.scheduler = default_scheduler
    elif args.sampler == "custom_euler":
        pipe.scheduler = create_custom_scheduler_for_model(
            model_key=model_key,
            device=device,
            fast_low_split_point=args.fast_low_split_point,
            fast_low_low_substeps=args.fast_low_low_substeps,
            low_region_noise_factor=args.low_region_noise_factor,
        )
    elif args.sampler in ("dpm", "custom_dpm"):
        pipe.scheduler = create_dpm_solver_scheduler(
            sampler=args.sampler,
            fast_low_split_point=args.fast_low_split_point,
            fast_low_low_substeps=args.fast_low_low_substeps,
            low_region_noise_factor=args.low_region_noise_factor,
        )
    else:
        pipe.scheduler = default_scheduler

    total_tasks = len(prompts_subset) * len(seeds)
    completed = 0
    skipped = 0

    for (line_number, prompt) in prompts_subset:
        # Apply prompt suffix if configured (e.g., for Qwen Image)
        actual_prompt = prompt + prompt_suffix if prompt_suffix else prompt

        for image_index, seed in enumerate(seeds, start=1):
            file_name = f"{line_number}_{image_index}.png"
            save_path = os.path.join(out_dir, file_name)

            # Auto-resume: skip if file already exists
            if auto_resume and os.path.exists(save_path):
                skipped += 1
                completed += 1
                continue

            generator = torch.Generator(device="cpu").manual_seed(seed)

            # Build pipeline call kwargs
            pipe_kwargs = {
                "prompt": actual_prompt,
                "num_inference_steps": num_inference_steps,
                "generator": generator,
                "height": height,
                "width": width,
            }

            # Add negative_prompt if configured (e.g., for Qwen Image)
            if negative_prompt:
                pipe_kwargs["negative_prompt"] = negative_prompt

            # Use true_cfg_scale for Qwen Image, guidance_scale for others
            if use_true_cfg:
                pipe_kwargs["true_cfg_scale"] = guidance_scale
            else:
                pipe_kwargs["guidance_scale"] = guidance_scale

            result = pipe(**pipe_kwargs)
            image = result.images[0]
            image.save(save_path)
            completed += 1
            print(f"[GPU {gpu_id}] {completed}/{total_tasks} done (skipped {skipped}): {save_path}")

    print(f"[GPU {gpu_id}] Finished model={model_key}, total={total_tasks}, skipped={skipped}")


def main() -> None:
    args = parse_args()
    ensure_directory(args.output_dir)

    model_registry = build_model_configs()
    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]

    for model_key in selected_models:
        if model_key not in model_registry:
            raise ValueError(f"Unknown model '{model_key}'. Valid: {', '.join(model_registry.keys())}")

    prompts_with_idx = load_prompts(args.prompts, start_line=args.start_line, end_line=args.end_line)

    if args.skip_empty:
        prompts_with_idx = [(i, p) for (i, p) in prompts_with_idx if p.strip() != ""]

    seeds = [0, 1000, 2000, 3000]
    height = args.resolution
    width = args.resolution

    # Parse GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(",") if g.strip()]
    else:
        # Auto-detect GPUs if device is cuda
        if args.device.startswith("cuda:"):
            gpu_ids = [int(args.device.split(":")[1])]
        elif args.device == "cuda":
            # Auto-detect all available GPUs
            num_gpus = torch.cuda.device_count()
            if num_gpus > 0:
                gpu_ids = list(range(num_gpus))
                print(f"[Auto-detect] Found {num_gpus} GPU(s): {gpu_ids}")
            else:
                gpu_ids = [0]
        else:
            gpu_ids = []  # CPU mode

    for model_key in selected_models:
        out_dir = get_output_dir(args, model_key)
        ensure_directory(out_dir)

        if args.dry_run:
            num_inference_steps = int(args.num_steps)
            print(f"[DRY-RUN] model={model_key}, steps={num_inference_steps}, resolution={width}x{height}, out_dir={out_dir}, gpus={gpu_ids}, prompts={len(prompts_with_idx)}")
            continue

        if not gpu_ids:
            # CPU mode: run single-threaded
            worker(
                gpu_id=0,
                args=args,
                model_key=model_key,
                prompts_subset=prompts_with_idx,
                seeds=seeds,
            )
        elif len(gpu_ids) == 1:
            # Single GPU mode
            worker(
                gpu_id=gpu_ids[0],
                args=args,
                model_key=model_key,
                prompts_subset=prompts_with_idx,
                seeds=seeds,
            )
        else:
            # Multi-GPU mode: split prompts across GPUs
            num_gpus = len(gpu_ids)
            prompts_per_gpu = len(prompts_with_idx) // num_gpus
            remainder = len(prompts_with_idx) % num_gpus

            processes = []
            start_idx = 0
            for i, gpu_id in enumerate(gpu_ids):
                # Distribute remainder prompts to first few GPUs
                end_idx = start_idx + prompts_per_gpu + (1 if i < remainder else 0)
                subset = prompts_with_idx[start_idx:end_idx]
                start_idx = end_idx

                if not subset:
                    continue

                p = Process(
                    target=worker,
                    args=(gpu_id, args, model_key, subset, seeds),
                )
                p.start()
                processes.append(p)
                print(f"[Main] Started GPU {gpu_id} with {len(subset)} prompts")

            # Wait for all processes to complete
            for p in processes:
                p.join()

            print(f"[Main] All GPUs finished for model={model_key}")


if __name__ == "__main__":
    # Use 'spawn' for CUDA multiprocessing compatibility
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
