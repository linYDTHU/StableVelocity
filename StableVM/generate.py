# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations
"""

import torch
import torch.distributed as dist
from models.sit import SiT_models
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from samplers import euler_sampler, euler_maruyama_sampler
from utils import load_legacy_checkpoints, download_model

def is_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def create_npz_from_sample_folder(sample_dir, num=None):
    """
    Builds a single .npz file from a folder of .png samples.

    - sample_dir: directory containing individual PNGs named like 000000.png, 000001.png, ...
    - num: optionally limit number of files to pack; default uses all found PNGs
    """
    png_files = []
    for fname in os.listdir(sample_dir):
        if not fname.lower().endswith(".png"):
            continue
        stem, _ = os.path.splitext(fname)
        try:
            idx = int(stem)
            png_files.append((idx, fname))
        except ValueError:
            # Skip non-numeric filenames
            continue
    if not png_files:
        raise FileNotFoundError(f"No .png files found in {sample_dir}")
    png_files.sort(key=lambda x: x[0])
    if num is not None:
        png_files = png_files[:num]

    samples = []
    for _, fname in tqdm(png_files, desc="Building .npz file from samples"):
        sample_pil = Image.open(os.path.join(sample_dir, fname))
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)

    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    use_ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1

    if use_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = 0 if torch.cuda.is_available() else "cpu"

    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    if args.unconditional:
        args.num_classes = 1
    else:
        args.num_classes = 1000

    # Load model:
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    latent_size = args.resolution // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=True,
        z_dims=[int(z_dim) for z_dim in args.projector_embed_dims.split(',')] if args.use_projector else None,
        encoder_depth=args.encoder_depth,
        use_projector=args.use_projector,
        use_irepa=args.use_irepa,
        **block_kwargs,
    ).to(device)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    # Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
    ckpt_path = args.ckpt
    if ckpt_path is None:
        args.ckpt = 'SiT-XL-2-256x256.pt'
        assert args.model == 'SiT-XL/2'
        assert len(args.projector_embed_dims.split(',')) == 1
        assert int(args.projector_embed_dims.split(',')[0]) == 768
        state_dict = download_model('last.pt')
    else:
        if device == "cpu":
            map_location = "cpu"
        else:
            map_location = f"cuda:{device}"
        state_dict = torch.load(ckpt_path, map_location=map_location, weights_only=False)['ema']
    if args.legacy:
        state_dict = load_legacy_checkpoints(
            state_dict=state_dict, encoder_depth=args.encoder_depth
            )
    model.load_state_dict(state_dict)
    model.eval()  # important!
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    
    # Base folder name
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.resolution}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}-{args.mode}-steps-{args.num_steps}-pred-{args.prediction_type}"
    
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * world_size
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        if args.use_projector:
            print(f"projector Parameters: {sum(p.numel() for p in model.projectors.parameters()):,}")
        print(f"Sampling mode: {args.mode}")
        print(f"Number of sampling steps: {args.num_steps}")
    
    # Build a global label schedule with exact counts, then (optionally) shuffle it.
    # IMPORTANT: all ranks must see the same permutation => use a rank-independent seed or broadcast.
    y_all = None
    if not args.unconditional and args.label_sampling == "equal":
        per_class = args.num_fid_samples // args.num_classes
        if rank == 0:
            y_all = torch.arange(args.num_classes, device=device).repeat_interleave(per_class)  # [0..999] each repeated per_class times
            # Pad to total_samples if needed
            if len(y_all) < total_samples:
                remaining = total_samples - len(y_all)
                y_all = torch.cat([y_all, torch.randint(0, args.num_classes, (remaining,), device=device)])
            gen = torch.Generator(device=device).manual_seed(args.global_seed)  # SAME seed across ranks
            y_all = y_all[torch.randperm(y_all.numel(), generator=gen, device=device)]
        
        # Broadcast the global label schedule to all ranks (only needed for DDP)
        if use_ddp:
            if rank != 0:
                y_all = torch.empty(total_samples, device=device, dtype=torch.long)
            dist.broadcast(y_all, src=0)
    elif not args.unconditional and args.label_sampling == "random":
        # Random sampling - labels will be generated per batch
        y_all = None
    elif args.unconditional:
        # Unconditional - no labels needed
        y_all = None
    else:
        raise NotImplementedError(f"Unknown label_sampling: {args.label_sampling}")

    # Calculate iterations needed
    iterations = int(math.ceil(total_samples / global_batch_size))
    
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    
    for iteration_idx, _ in enumerate(pbar):
        # Calculate how many samples to generate in this iteration
        samples_left = total_samples - (iteration_idx * global_batch_size)
        if samples_left <= 0:
            break
            
        current_batch_size = min(n, samples_left - rank * n) if samples_left < global_batch_size else n
        if current_batch_size <= 0:
            continue
            
        # Sample inputs:
        z = torch.randn(current_batch_size, model.in_channels, latent_size, latent_size, device=device)
        if args.unconditional:
            y = None
        elif y_all is not None:
            # Use pre-generated labels from the global schedule
            # Each rank processes its portion of the current batch
            start_idx = iteration_idx * global_batch_size + rank * n
            end_idx = min(start_idx + current_batch_size, len(y_all))
            y = y_all[start_idx:end_idx]
            # Pad if needed (shouldn't happen, but safety check)
            if len(y) < current_batch_size:
                padding = torch.randint(0, args.num_classes, (current_batch_size - len(y),), device=device)
                y = torch.cat([y, padding])
        else:
            # Random sampling per batch
            y = torch.randint(0, args.num_classes, (current_batch_size,), device=device)

        # Sample images:
        sampling_kwargs = dict(
            model=model, 
            latents=z,
            y=y,
            num_steps=args.num_steps, 
            heun=args.heun,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            path_type=args.path_type,
            edm_schedule=args.edm_schedule,
            prediction_type=args.prediction_type,
        )

        with torch.no_grad():
            if args.mode == "sde":
                samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
            elif args.mode == "ode":
                samples = euler_sampler(**sampling_kwargs).to(torch.float32)
            else:
                raise NotImplementedError(f"Unknown sampling mode: {args.mode}")

            latents_scale = torch.tensor(
                [0.18215, 0.18215, 0.18215, 0.18215, ]
                ).view(1, 4, 1, 1).to(device)
            latents_bias = -torch.tensor(
                [0., 0., 0., 0.,]
                ).view(1, 4, 1, 1).to(device)
            samples = vae.decode((samples -  latents_bias) / latents_scale).sample
            samples = (samples + 1) / 2.
            samples = torch.clamp(
                255. * samples, 0, 255
                ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                if i >= current_batch_size:  # Safety check
                    break
                index = iteration_idx * global_batch_size + i * world_size + rank
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    if use_ddp:
        dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, num=args.num_fid_samples)
        print("Done.")
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=0)

    # precision
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    # logging/saving:
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a SiT checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="samples")

    # model
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-projector", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-irepa", action=argparse.BooleanOptionalAction, default=False,
                        help="Use iREPA improvements: Conv projector instead of MLP.")

    # vae
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")

    # number of samples
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)

    # sampling related hyperparameters
    parser.add_argument("--unconditional", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--label-sampling", type=str, default="random", choices=["equal", "random"],
                        help="Label sampling strategy: 'equal' ensures each class appears equally, 'random' samples uniformly")
    parser.add_argument("--mode", type=str, default="ode", choices=["ode", "sde"], 
                        help="Sampling mode: ode (Euler ODE) or sde (Euler-Maruyama SDE)")
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--projector-embed-dims", type=str, default="768")
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False) # only for ode
    parser.add_argument("--guidance-low", type=float, default=0.)
    parser.add_argument("--guidance-high", type=float, default=1.)
    parser.add_argument("--edm-schedule", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prediction-type", type=str, default="v", choices=["v", "x"])
    
    # will be deprecated
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False) # only for ode

    args = parser.parse_args()
    main(args)
