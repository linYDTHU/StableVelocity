#!/usr/bin/env python3
"""
Text-to-Image Demo: StableVS Scheduler Comparison

Generates images using SD3.5-Large, Flux, and Qwen-Image models,
comparing baseline Euler scheduling against StableVS accelerated sampling.

For each model the script produces three images:
  1. Euler / default scheduler with 30 steps  (baseline quality)
  2. Euler / default scheduler with 20 steps  (reduced-step baseline)
  3. StableVS scheduler with 20 total steps   (11 Euler + 9 StableVS)

Usage:
    # Generate with all three models (requires ~80 GB VRAM for sequential loading)
    python t2i_demo.py --models flux,sd35,qwen --output-dir ./figures

    # Generate with a single model
    python t2i_demo.py --models sd35 --output-dir ./figures

    # Print sigma schedules only (no GPU needed)
    python t2i_demo.py --print-sigmas-only
"""

import argparse
import os
from typing import Optional

import torch

from diffusers import DiffusionPipeline, FluxPipeline, StableDiffusion3Pipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

# Import custom schedulers from stablevs
from stablevs import StableVSFlowMatchScheduler


# ---------------------------------------------------------------------------
# Default prompts used in the paper
# ---------------------------------------------------------------------------
PROMPTS = {
    "sd35": (
        "A turquoise river winds through a lush canyon. Thick moss and dense ferns "
        "blanket the rocky walls; multiple waterfalls cascade from above, enveloped "
        "in mist. At noon, sunlight filters through the dense canopy, dappling the "
        "river surface with shimmering light. The atmosphere is humid and fresh, "
        "pulsing with primal jungle vitality. No humans, text, or artificial traces present."
    ),
    "flux": "A cat holding a sign that says Stable Velocity",
    "qwen": (
        "A 20-year-old East Asian girl with delicate, charming features and large, "
        "bright brown eyes—expressive and lively, with a cheerful or subtly smiling "
        "expression. Her naturally wavy long hair is either loose or tied in twin "
        "ponytails. She has fair skin and light makeup accentuating her youthful "
        "freshness. She wears a modern, cute dress or relaxed outfit in bright, soft "
        "colors—lightweight fabric, minimalist cut. She stands indoors at an anime "
        "convention, surrounded by banners, posters, or stalls. Lighting is typical "
        "indoor illumination—no staged lighting—and the image resembles a casual "
        "iPhone snapshot: unpretentious composition, yet brimming with vivid, fresh, "
        "youthful charm."
    ),
}

QWEN_NEGATIVE_PROMPT = (
    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，"
    "人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Compute dynamic-shifting mu (same formula as in Flux / Qwen pipelines)."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


# ---------------------------------------------------------------------------
# Sigma schedule printing
# ---------------------------------------------------------------------------
def print_sigmas_comparison(args: argparse.Namespace) -> None:
    """Print sigma schedules for all scheduler configurations."""

    print("\n" + "=" * 80)
    print("SD3.5 / EULER SCHEDULER SIGMAS COMPARISON")
    print("=" * 80)

    for n_steps in (30, 20):
        sched = FlowMatchEulerDiscreteScheduler(shift=3.0, num_train_timesteps=1000)
        sched.set_timesteps(n_steps)
        print(f"\n[Euler {n_steps} steps] sigmas ({len(sched.sigmas)}):")
        print(sched.sigmas)
        print(f"[Euler {n_steps} steps] timesteps ({len(sched.timesteps)}):")
        print(sched.timesteps)

    stablevs_sched = StableVSFlowMatchScheduler(
        shift=3.0,
        num_train_timesteps=1000,
        use_fast_low_schedule=True,
        fast_low_split_point=args.fast_low_split_point,
        fast_low_low_substeps=args.fast_low_low_substeps,
        low_region_noise_factor=args.low_region_noise_factor,
    )
    stablevs_sched.set_timesteps(args.num_steps)
    print(f"\n[StableVS {args.num_steps} steps] sigmas ({len(stablevs_sched.sigmas)}):")
    print(stablevs_sched.sigmas)
    print(f"[StableVS {args.num_steps} steps] timesteps ({len(stablevs_sched.timesteps)}):")
    print(stablevs_sched.timesteps)

    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FLUX SCHEDULER SIGMAS (with dynamic shifting)")
    print("=" * 80)

    flux_mu = calculate_shift(4096, 256, 4096, 0.5, 1.15)
    print(f"\n[Flux] image_seq_len=4096, computed mu={flux_mu:.4f}")

    for n_steps in (30, 20):
        sched = FlowMatchEulerDiscreteScheduler(
            base_image_seq_len=256, max_image_seq_len=4096,
            use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15,
            shift=3.0, num_train_timesteps=1000,
        )
        sched.set_timesteps(n_steps, mu=flux_mu)
        print(f"\n[Flux Euler {n_steps} steps] sigmas ({len(sched.sigmas)}):")
        print(sched.sigmas)
        print(f"[Flux Euler {n_steps} steps] timesteps ({len(sched.timesteps)}):")
        print(sched.timesteps)

    flux_sv = StableVSFlowMatchScheduler(
        base_image_seq_len=256, max_image_seq_len=4096,
        use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15,
        shift=3.0, num_train_timesteps=1000,
        use_fast_low_schedule=True,
        fast_low_split_point=args.fast_low_split_point,
        fast_low_low_substeps=args.fast_low_low_substeps,
        low_region_noise_factor=args.low_region_noise_factor,
    )
    flux_sv.set_timesteps(args.num_steps, mu=flux_mu)
    print(f"\n[Flux StableVS {args.num_steps} steps] sigmas ({len(flux_sv.sigmas)}):")
    print(flux_sv.sigmas)
    print(f"[Flux StableVS {args.num_steps} steps] timesteps ({len(flux_sv.timesteps)}):")
    print(flux_sv.timesteps)

    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("QWEN SCHEDULER SIGMAS (with exponential time shift)")
    print("=" * 80)

    qwen_mu = calculate_shift(4096, 256, 8192, 0.5, 0.9)
    print(f"\n[Qwen] image_seq_len=4096, computed mu={qwen_mu:.4f}")

    qwen_cfg = dict(
        base_image_seq_len=256, max_image_seq_len=8192,
        use_dynamic_shifting=True, base_shift=0.5, max_shift=0.9,
        shift=1.0, shift_terminal=0.02, time_shift_type="exponential",
        num_train_timesteps=1000,
    )

    for n_steps in (30, 17):
        sched = FlowMatchEulerDiscreteScheduler(**qwen_cfg)
        sched.set_timesteps(n_steps, mu=qwen_mu)
        print(f"\n[Qwen Euler {n_steps} steps] sigmas ({len(sched.sigmas)}):")
        print(sched.sigmas)
        print(f"[Qwen Euler {n_steps} steps] timesteps ({len(sched.timesteps)}):")
        print(sched.timesteps)

    qwen_sv = StableVSFlowMatchScheduler(
        **qwen_cfg,
        use_fast_low_schedule=True,
        fast_low_split_point=args.fast_low_split_point,
        fast_low_low_substeps=args.fast_low_low_substeps,
        low_region_noise_factor=args.low_region_noise_factor,
    )
    qwen_sv.set_timesteps(args.num_steps, mu=qwen_mu)
    print(f"\n[Qwen StableVS {args.num_steps} steps] sigmas ({len(qwen_sv.sigmas)}):")
    print(qwen_sv.sigmas)
    print(f"[Qwen StableVS {args.num_steps} steps] timesteps ({len(qwen_sv.timesteps)}):")
    print(qwen_sv.timesteps)

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------
def generate_sd35(args: argparse.Namespace) -> None:
    """Generate images with Stable Diffusion 3.5 Large."""
    prompt = PROMPTS["sd35"]
    out = args.output_dir

    print("\n" + "=" * 80)
    print("SD3.5-Large — Generating images")
    print("=" * 80)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16
    ).to("cuda")

    # 1. Euler 30 steps
    pipe.scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0, num_train_timesteps=1000)
    image = pipe(
        prompt, num_inference_steps=30, guidance_scale=4.5,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]
    image.save(os.path.join(out, "sd35_euler_30steps.jpg"))
    print("Saved: sd35_euler_30steps.jpg")

    # 2. Euler 20 steps
    pipe.scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0, num_train_timesteps=1000)
    image = pipe(
        prompt, num_inference_steps=20, guidance_scale=4.5,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]
    image.save(os.path.join(out, "sd35_euler_20steps.jpg"))
    print("Saved: sd35_euler_20steps.jpg")

    # 3. StableVS 20 steps (Euler 11 + StableVS 9)
    pipe.scheduler = StableVSFlowMatchScheduler(
        shift=3.0, num_train_timesteps=1000,
        use_fast_low_schedule=True,
        fast_low_split_point=args.fast_low_split_point,
        fast_low_low_substeps=args.fast_low_low_substeps,
        low_region_noise_factor=args.low_region_noise_factor,
    )
    image = pipe(
        prompt, num_inference_steps=args.num_steps, guidance_scale=4.5,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]
    image.save(os.path.join(out, f"sd35_stablevs_{args.num_steps}steps.jpg"))
    print(f"Saved: sd35_stablevs_{args.num_steps}steps.jpg")

    print("--- SD3.5 done ---\n")
    del pipe
    torch.cuda.empty_cache()


def generate_flux(args: argparse.Namespace) -> None:
    """Generate images with FLUX.1-dev."""
    prompt = PROMPTS["flux"]
    out = args.output_dir

    print("\n" + "=" * 80)
    print("FLUX.1-dev — Generating images")
    print("=" * 80)

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    ).to("cuda")

    # 1. Euler 30 steps (default scheduler with dynamic shifting)
    image = pipe(
        prompt, num_inference_steps=30, guidance_scale=3.5,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]
    image.save(os.path.join(out, "flux_euler_30steps.jpg"))
    print("Saved: flux_euler_30steps.jpg")

    # 2. Euler 20 steps
    image = pipe(
        prompt, num_inference_steps=20, guidance_scale=3.5,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]
    image.save(os.path.join(out, "flux_euler_20steps.jpg"))
    print("Saved: flux_euler_20steps.jpg")

    # 3. StableVS 20 steps
    pipe.scheduler = StableVSFlowMatchScheduler(
        base_image_seq_len=256, max_image_seq_len=4096,
        use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15,
        shift=3.0, num_train_timesteps=1000,
        use_fast_low_schedule=True,
        fast_low_split_point=args.fast_low_split_point,
        fast_low_low_substeps=args.fast_low_low_substeps,
        low_region_noise_factor=args.low_region_noise_factor,
    )
    image = pipe(
        prompt, num_inference_steps=args.num_steps, guidance_scale=3.5,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]
    image.save(os.path.join(out, f"flux_stablevs_{args.num_steps}steps.jpg"))
    print(f"Saved: flux_stablevs_{args.num_steps}steps.jpg")

    print("--- Flux done ---\n")
    del pipe
    torch.cuda.empty_cache()


def generate_qwen(args: argparse.Namespace) -> None:
    """Generate images with Qwen-Image-2512."""
    prompt = PROMPTS["qwen"]
    neg = QWEN_NEGATIVE_PROMPT
    out = args.output_dir

    print("\n" + "=" * 80)
    print("Qwen-Image-2512 — Generating images")
    print("=" * 80)

    pipe = DiffusionPipeline.from_pretrained(
        "Qwen/Qwen-Image-2512", torch_dtype=torch.bfloat16
    ).to("cuda")

    # 1. Euler 30 steps (default scheduler with exponential time shift)
    image = pipe(
        prompt, negative_prompt=neg, num_inference_steps=30,
        true_cfg_scale=4.0,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]
    image.save(os.path.join(out, "qwen_euler_30steps.jpg"))
    print("Saved: qwen_euler_30steps.jpg")

    # 2. Euler 17 steps
    image = pipe(
        prompt, negative_prompt=neg, num_inference_steps=17,
        true_cfg_scale=4.0,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]
    image.save(os.path.join(out, "qwen_euler_17steps.jpg"))
    print("Saved: qwen_euler_17steps.jpg")

    # 3. StableVS 17 steps (Euler 8 + StableVS 9)
    pipe.scheduler = StableVSFlowMatchScheduler(
        base_image_seq_len=256, max_image_seq_len=8192,
        use_dynamic_shifting=True, base_shift=0.5, max_shift=0.9,
        shift=1.0, shift_terminal=0.02, time_shift_type="exponential",
        num_train_timesteps=1000,
        use_fast_low_schedule=True,
        fast_low_split_point=args.fast_low_split_point,
        fast_low_low_substeps=args.fast_low_low_substeps,
        low_region_noise_factor=args.low_region_noise_factor,
    )
    image = pipe(
        prompt, negative_prompt=neg, num_inference_steps=args.num_steps,
        true_cfg_scale=4.0,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]
    image.save(os.path.join(out, f"qwen_stablevs_{args.num_steps}steps.jpg"))
    print(f"Saved: qwen_stablevs_{args.num_steps}steps.jpg")

    print("--- Qwen done ---\n")
    del pipe
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
MODEL_GENERATORS = {
    "sd35": generate_sd35,
    "flux": generate_flux,
    "qwen": generate_qwen,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Text-to-Image demo comparing Euler vs StableVS schedulers."
    )
    parser.add_argument(
        "--models", type=str, default="sd35,flux,qwen",
        help="Comma-separated model list: sd35, flux, qwen (default: all).",
    )
    parser.add_argument("--output-dir", type=str, default="figures", help="Output directory.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num-steps", type=int, default=30,
                        help="Total inference steps for StableVS scheduler (default: 30).")
    parser.add_argument("--print-sigmas-only", action="store_true",
                        help="Print sigma schedules and exit (no generation).")
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

    os.makedirs(args.output_dir, exist_ok=True)

    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    for model_key in selected:
        if model_key not in MODEL_GENERATORS:
            raise ValueError(
                f"Unknown model '{model_key}'. Choose from: {', '.join(MODEL_GENERATORS)}"
            )
        MODEL_GENERATORS[model_key](args)

    print("\nAll done!")


if __name__ == "__main__":
    main()
