#!/usr/bin/env python3
"""
Text-to-Video Demo: StableVS Scheduler Comparison

Generates videos using the Wan2.2-T2V-A14B model, comparing baseline
UniPC scheduling against StableVS accelerated sampling.

For each prompt the script produces three videos:
  1. UniPC scheduler with 30 steps  (baseline quality)
  2. UniPC scheduler with 20 steps  (reduced-step baseline)
  3. StableVS-UniPC with 20 total steps (UniPC 11 + StableVS 9)

Usage:
    # Generate with default prompts
    python t2v_demo.py --output-dir ./videos

    # Print sigma schedules only (no GPU needed)
    python t2v_demo.py --print-sigmas-only

    # Custom prompt
    python t2v_demo.py --prompt "A horse jumps over a fence." --output-dir ./videos
"""

import argparse
import os
from typing import Optional

import torch

from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

# Import custom scheduler from stablevs
from stablevs import StableVSUniPCMultistepScheduler


# ---------------------------------------------------------------------------
# Default prompts used in the paper
# ---------------------------------------------------------------------------
DEFAULT_PROMPTS = [
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    "A horse jumps over a fence.",
]

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


# ---------------------------------------------------------------------------
# Shared UniPC base configuration (matches Wan2.2 defaults)
# ---------------------------------------------------------------------------
UNIPC_BASE_CFG = dict(
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
    timestep_spacing="linspace",
    trained_betas=None,
    use_beta_sigmas=False,
    use_exponential_sigmas=False,
    use_flow_sigmas=True,
    use_karras_sigmas=False,
)


# ---------------------------------------------------------------------------
# Sigma schedule printing
# ---------------------------------------------------------------------------
def print_sigmas_comparison(args: argparse.Namespace) -> None:
    """Print sigma schedules for UniPC vs StableVS-UniPC."""

    print("\n" + "=" * 80)
    print("UNIPC SCHEDULER SIGMAS COMPARISON")
    print("=" * 80)

    # UniPC 30 steps
    unipc_30 = UniPCMultistepScheduler(**UNIPC_BASE_CFG)
    unipc_30.set_timesteps(30)
    print(f"\n[UniPC 30 steps] sigmas ({len(unipc_30.sigmas)}):")
    print(unipc_30.sigmas)
    print(f"[UniPC 30 steps] timesteps ({len(unipc_30.timesteps)}):")
    print(unipc_30.timesteps)

    # UniPC 20 steps
    unipc_20 = UniPCMultistepScheduler(**UNIPC_BASE_CFG)
    unipc_20.set_timesteps(20)
    print(f"\n[UniPC 20 steps] sigmas ({len(unipc_20.sigmas)}):")
    print(unipc_20.sigmas)
    print(f"[UniPC 20 steps] timesteps ({len(unipc_20.timesteps)}):")
    print(unipc_20.timesteps)

    # StableVS-UniPC
    stablevs_unipc = StableVSUniPCMultistepScheduler(
        **UNIPC_BASE_CFG,
        time_shift_type="exponential",
        use_dynamic_shifting=False,
        use_fast_low_schedule=True,
        fast_low_split_point=args.fast_low_split_point,
        fast_low_low_substeps=args.fast_low_low_substeps,
        low_region_noise_factor=args.low_region_noise_factor,
    )
    stablevs_unipc.set_timesteps(args.num_steps)
    print(f"\n[StableVS-UniPC {args.num_steps} steps] sigmas ({len(stablevs_unipc.sigmas)}):")
    print(stablevs_unipc.sigmas)
    print(f"[StableVS-UniPC {args.num_steps} steps] timesteps ({len(stablevs_unipc.timesteps)}):")
    print(stablevs_unipc.timesteps)

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------
def generate_videos(args: argparse.Namespace) -> None:
    """Generate comparison videos with Wan2.2."""
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    dtype = torch.bfloat16
    device = "cuda"

    # Load model
    print("Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        args.model_id, subfolder="vae", torch_dtype=torch.float32
    )
    print("Loading pipeline...")
    pipe = WanPipeline.from_pretrained(
        args.model_id, vae=vae, torch_dtype=dtype
    )
    pipe.to(device)

    if args.cpu_offload:
        pipe.enable_model_cpu_offload()

    # Build StableVS scheduler
    stablevs_scheduler = StableVSUniPCMultistepScheduler(
        **UNIPC_BASE_CFG,
        time_shift_type="exponential",
        use_dynamic_shifting=False,
        use_fast_low_schedule=True,
        fast_low_split_point=args.fast_low_split_point,
        fast_low_low_substeps=args.fast_low_low_substeps,
        low_region_noise_factor=args.low_region_noise_factor,
    )

    # Store the default scheduler for restoration
    default_scheduler = pipe.scheduler

    # Determine prompts to generate
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = DEFAULT_PROMPTS

    common_kwargs = dict(
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        guidance_scale_2=args.guidance_scale_2,
    )

    for p_idx, prompt in enumerate(prompts):
        # Create a short tag for filenames
        tag = prompt[:40].replace(" ", "_").replace(".", "").lower()
        tag = "".join(c for c in tag if c.isalnum() or c == "_")

        print(f"\n{'='*60}")
        print(f"Prompt {p_idx + 1}/{len(prompts)}: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"{'='*60}")

        # 1. UniPC 30 steps (baseline)
        pipe.scheduler = default_scheduler
        print(f"\n  [1/3] UniPC 30 steps ...")
        output = pipe(
            prompt=prompt, num_inference_steps=30,
            generator=torch.Generator("cpu").manual_seed(args.seed),
            **common_kwargs,
        ).frames[0]
        path_30 = os.path.join(out, f"{tag}_unipc_30steps.mp4")
        export_to_video(output, path_30, fps=args.fps)
        print(f"  Saved: {path_30}")

        # 2. UniPC 20 steps (reduced-step baseline)
        pipe.scheduler = default_scheduler
        print(f"  [2/3] UniPC 20 steps ...")
        output = pipe(
            prompt=prompt, num_inference_steps=20,
            generator=torch.Generator("cpu").manual_seed(args.seed),
            **common_kwargs,
        ).frames[0]
        path_20 = os.path.join(out, f"{tag}_unipc_20steps.mp4")
        export_to_video(output, path_20, fps=args.fps)
        print(f"  Saved: {path_20}")

        # 3. StableVS-UniPC (accelerated)
        pipe.scheduler = stablevs_scheduler
        print(f"  [3/3] StableVS-UniPC {args.num_steps} steps ...")
        output = pipe(
            prompt=prompt, num_inference_steps=args.num_steps,
            generator=torch.Generator("cpu").manual_seed(args.seed),
            **common_kwargs,
        ).frames[0]
        path_sv = os.path.join(out, f"{tag}_stablevs_{args.num_steps}steps.mp4")
        export_to_video(output, path_sv, fps=args.fps)
        print(f"  Saved: {path_sv}")

    print("\nAll videos generated!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Text-to-Video demo comparing UniPC vs StableVS-UniPC schedulers on Wan2.2."
    )
    parser.add_argument("--output-dir", type=str, default="videos", help="Output directory.")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt. If not set, uses built-in demo prompts.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num-steps", type=int, default=30,
                        help="Total inference steps for StableVS scheduler (default: 30).")
    parser.add_argument("--print-sigmas-only", action="store_true",
                        help="Print sigma schedules and exit (no generation).")
    # Model
    parser.add_argument("--model-id", type=str, default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                        help="Wan model ID.")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Enable model CPU offload to reduce VRAM usage.")
    # Video parameters
    parser.add_argument("--height", type=int, default=480, help="Video frame height.")
    parser.add_argument("--width", type=int, default=640, help="Video frame width.")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames per video.")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second for saved videos.")
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="Guidance scale.")
    parser.add_argument("--guidance-scale-2", type=float, default=3.0,
                        help="Second guidance scale for Wan2.2.")
    parser.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT,
                        help="Negative prompt.")
    # StableVS parameters
    parser.add_argument("--fast-low-split-point", type=float, default=0.85,
                        help="Split point between high and low variance regions (default: 0.85).")
    parser.add_argument("--fast-low-low-substeps", type=int, default=9,
                        help="Number of substeps in the low-variance region (default: 9).")
    parser.add_argument("--low-region-noise-factor", type=float, default=0.0,
                        help="Noise factor for the low-variance region (default: 0.0).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Always print sigma comparison
    print_sigmas_comparison(args)

    if args.print_sigmas_only:
        return

    generate_videos(args)


if __name__ == "__main__":
    main()
