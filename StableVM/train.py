import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.sit import SiT_models
from loss import SILoss, StableVMLoss, MemoryBank
from utils import load_encoders

from dataset import CustomDataset


class IndexedDataset(torch.utils.data.Dataset):
    """Wrapper that returns (original_data..., index) for any dataset."""
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        if isinstance(data, tuple):
            return (*data, idx)
        else:
            return (data, idx)

from diffusers.models import AutoencoderKL
import wandb
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


def irepa_spatial_normalize(x, gamma=1.0):
    """
    Apply iREPA spatial normalization to encoder features.
    
    Args:
        x: Encoder features with shape [B, T, D] where T is the spatial/token dimension
        gamma: Coefficient for mean subtraction (default: 1.0)
    
    Returns:
        Spatially normalized features with the same shape [B, T, D]
    """
    # x: [B, T, D]
    # Subtract gamma * spatial mean
    x = x - gamma * x.mean(dim=1, keepdim=True)
    # Normalize by spatial std
    x = x / (x.std(dim=1, keepdim=True) + 1e-6)
    return x

def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):    
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z 


@torch.no_grad()
def update_ema(ema_model, model, num_updates, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    # update the decay rate according to https://github.com/facebookresearch/flow_matching/blob/main/examples/image/models/ema.py
    decay = min(decay, (1 + num_updates) / (10 + num_updates))

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    
    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    use_cfg = (args.cfg_prob > 0)
    
    # Load encoders if using projection loss (for either SI or StableVM loss)
    if args.use_proj_loss and args.enc_type != None:
        # Load encoders on main process first to avoid race conditions in torch.hub cache
        if accelerator.is_main_process:
            encoders, encoder_types, architectures = load_encoders(
                args.enc_type, device, args.resolution
            )
        accelerator.wait_for_everyone()
        # Now other processes can safely load from cache
        if not accelerator.is_main_process:
            encoders, encoder_types, architectures = load_encoders(
                args.enc_type, device, args.resolution
            )
        z_dims = [encoder.embed_dim for encoder in encoders]
    else:
        encoders, encoder_types, architectures = None, None, None
        z_dims = [0]  # Default z_dims for models without encoders
    
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=use_cfg,
        z_dims=z_dims,
        encoder_depth=args.encoder_depth,
        use_projector=args.use_proj_loss,
        use_irepa=args.use_irepa,
        **block_kwargs
    )
    model = model.to(device)
    ema = deepcopy(model).to(device)    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
    requires_grad(ema, False)
    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
    ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
    ).view(1, 4, 1, 1).to(device)

    # Setup data
    base_dataset = CustomDataset(args.data_dir)

    # create loss function
    if args.loss_type == "si":
        loss_fn = SILoss(
            prediction=args.prediction,
            path_type=args.path_type, 
            encoders=encoders,
            accelerator=accelerator,
            latents_scale=latents_scale,
            latents_bias=latents_bias,
            weighting=args.weighting,
            use_proj_loss=args.use_proj_loss,
            proj_weight_schedule=args.proj_weight_schedule,
            proj_tau=args.proj_tau,
            proj_k=args.proj_k,
        )
    elif args.loss_type == "stablevm":
        local_batch_size = int(args.batch_size // accelerator.num_processes)
        loss_fn = StableVMLoss(
            batch_size=local_batch_size,
            num_classes=args.num_classes,
            prediction=args.prediction,
            path_type=args.path_type,
            weighting=args.weighting,
            cfg_prob=args.cfg_prob,
            bank_chunk_size=args.bank_chunk_size,
            use_proj_loss=args.use_proj_loss,
            proj_weight_schedule=args.proj_weight_schedule,
            proj_tau=args.proj_tau,
            proj_k=args.proj_k,
            encoders=encoders if args.use_proj_loss else None,
            encoder_types=encoder_types if args.use_proj_loss else None,
            preprocess_fn=preprocess_raw_image if args.use_proj_loss else None,
            dataset=base_dataset if args.use_proj_loss else None,  # Use base dataset for index-based loading
            use_irepa=args.use_irepa,
            irepa_gamma=args.irepa_gamma,
        )
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    
    # Wrap dataset to include indices if needed for StableVM with projection loss
    if args.loss_type == "stablevm" and args.use_proj_loss:
        train_dataset = IndexedDataset(base_dataset)
    else:
        train_dataset = base_dataset
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    
    # Memory bank for StableVMLoss
    memory_bank = None
    if args.loss_type == "stablevm":
        feature_shape = (4, latent_size, latent_size)
        # Keep memory bank on CPU (pinned) to save GPU RAM, store as fp16 for memory efficiency
        # Store dataset indices for on-the-fly image loading (extremely memory efficient: ~2MB total)
        # Choose memory bank device based on args
        bank_device = device if args.bank_on_gpu else 'cpu'
        memory_bank = MemoryBank(
            num_classes=args.num_classes,
            capacity_per_class=args.bank_capacity_per_class,
            feature_shape=feature_shape,
            device=bank_device,  # GPU: faster (no transfer), CPU: saves VRAM
            dtype=torch.float16,  # Storage in fp16 to halve memory usage
            pin_memory=(not args.bank_on_gpu),  # Pinned memory only for CPU
            store_indices=args.use_proj_loss,
        )
 
    # Prepare models for training:
    update_ema(ema, model, 0, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            weights_only=False
            )
        model.load_state_dict(ckpt['model'])
        if 'ema' in ckpt:
            ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # Prefill memory bank before training starts (for StableVMLoss)
    if memory_bank is not None:
        if accelerator.is_main_process:
            logger.info("Prefilling memory bank with training data (per process shard)...")
        with torch.no_grad():
            if args.prefill_bank_fully:
                # Use a larger, dedicated prefill dataloader for speed
                prefill_workers = args.prefill_workers if args.prefill_workers is not None else args.num_workers
                prefill_loader = DataLoader(
                    train_dataset,
                    batch_size=args.prefill_batch_size,
                    shuffle=True,
                    num_workers=prefill_workers,
                    pin_memory=True,
                    drop_last=True,
                )
                prefill_loader = accelerator.prepare(prefill_loader)
                for prefill_step, batch in enumerate(prefill_loader):
                    if args.use_proj_loss:
                        raw_image, x, y, indices = batch[0], batch[1], batch[2], batch[3]
                    else:
                        raw_image, x, y = batch[0], batch[1], batch[2]
                        indices = None
                    raw_image = raw_image.to(device)
                    x = x.squeeze(dim=1).to(device)
                    y = y.to(device)
                    feats_for_bank = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
                    memory_bank.add(feats_for_bank, y, dataset_indices=indices)

                    all_full = all(memory_bank.data[c].shape[0] >= args.bank_capacity_per_class for c in range(args.num_classes))
                    if all_full:
                        break
            else:
                # Minimal prefill: stop once each class has at least one entry
                for prefill_step, batch in enumerate(train_dataloader):
                    if args.use_proj_loss:
                        raw_image, x, y, indices = batch[0], batch[1], batch[2], batch[3]
                    else:
                        raw_image, x, y = batch[0], batch[1], batch[2]
                        indices = None
                    raw_image = raw_image.to(device)
                    x = x.squeeze(dim=1).to(device)
                    y = y.to(device)
                    feats_for_bank = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
                    memory_bank.add(feats_for_bank, y, dataset_indices=indices)

                    all_have_any = all(memory_bank.data[c].shape[0] >= 1 for c in range(args.num_classes))
                    if all_have_any:
                        break
        if accelerator.is_main_process:
            filled_counts = {c: int(memory_bank.data[c].shape[0]) for c in range(args.num_classes)}
            logger.info(f"Memory bank prefilling complete. Per-class sizes (first 10): {list(filled_counts.items())[:10]} ...")
 
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="Stable_VM", 
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )
        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 64 // accelerator.num_processes
    sample_batch = next(iter(train_dataloader))
    gt_raw_images, gt_xs = sample_batch[0], sample_batch[1]
    assert gt_raw_images.shape[-1] == args.resolution
    gt_xs = gt_xs[:sample_batch_size]
    gt_xs = sample_posterior(
        gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
    )
    ys = torch.randint(args.num_classes, size=(sample_batch_size,), device=device)
    # Create sampling noise:
    n = ys.size(0)
    xT = torch.randn((n, 4, latent_size, latent_size), device=device)
        
    for epoch in range(args.epochs):
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            if args.loss_type == "stablevm" and args.use_proj_loss:
                raw_image, x, y, batch_indices = batch[0], batch[1], batch[2], batch[3]
            else:
                raw_image, x, y = batch[0], batch[1], batch[2]
                batch_indices = None
            raw_image = raw_image.to(device)
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)
            
            if args.loss_type == "stablevm":
                labels = y
            else:
                drop_ids = torch.rand(y.shape[0], device=y.device) < args.cfg_prob
                labels = torch.where(drop_ids, args.num_classes, y)
            
            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
                # Process encoder features if using SI loss with projection loss
                # (StableVMLoss computes features on-the-fly from raw images in memory bank)
                if args.loss_type == "si" and args.use_proj_loss and encoders is not None:
                    zs = []
                    with accelerator.autocast():
                        for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                            raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                            z = encoder.forward_features(raw_image_)
                            if 'mocov3' in encoder_type: z = z[:, 1:] 
                            if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                            # Apply iREPA spatial normalization if enabled
                            if args.use_irepa:
                                z = irepa_spatial_normalize(z, gamma=args.irepa_gamma)
                            zs.append(z)
                else:
                    zs = None

            # update memory bank with current batch
            if args.loss_type == "stablevm" and memory_bank is not None:
                # Pass dataset indices for on-the-fly image loading
                memory_bank.add(x, y, dataset_indices=batch_indices)

            with accelerator.accumulate(model):
                model_kwargs = dict(y=labels)
                if args.loss_type == "stablevm":
                    # New StableVM objective samples from the memory bank internally
                    if args.use_proj_loss:
                        loss, proj_loss = loss_fn(model, memory_bank=memory_bank, device=device)
                        loss_mean = loss.mean()
                        proj_loss_mean = proj_loss.mean() if proj_loss is not None else torch.tensor(0., device=device)
                        loss = loss_mean + proj_loss_mean * args.proj_coeff
                    else:
                        loss = loss_fn(model, memory_bank=memory_bank, device=device)
                        loss_mean = loss.mean()
                        loss = loss_mean
                elif args.loss_type == "si":
                    if not args.use_proj_loss:
                        loss = loss_fn(model, x, model_kwargs)
                        loss_mean = loss.mean()
                        loss = loss_mean
                    else:
                        loss, proj_loss = loss_fn(model, x, model_kwargs, zs=zs)
                        loss_mean = loss.mean()
                        proj_loss_mean = proj_loss.mean()
                        loss = loss_mean + proj_loss_mean * args.proj_coeff
                    
                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model, num_updates=global_step, decay=args.ema_decay)
            
            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1                
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                from samplers import euler_sampler
                with torch.no_grad():
                    samples = euler_sampler(
                        ema, 
                        xT, 
                        ys,
                        num_steps=50, 
                        cfg_scale=4.0,
                        guidance_low=0.,
                        guidance_high=1.,
                        path_type=args.path_type,
                        heun=False,
                        edm_schedule=False,
                    ).to(torch.float32)
                    samples = vae.decode((samples -  latents_bias) / latents_scale).sample
                    gt_samples = vae.decode((gt_xs - latents_bias) / latents_scale).sample
                    samples = (samples + 1) / 2.
                    gt_samples = (gt_samples + 1) / 2.
                out_samples = accelerator.gather(samples.to(torch.float32))
                gt_samples = accelerator.gather(gt_samples.to(torch.float32))
                accelerator.log({"samples": wandb.Image(array2grid(out_samples)),
                                 "gt_samples": wandb.Image(array2grid(gt_samples))})
                logging.info("Generating EMA samples done.")

            logs = {
                "loss": accelerator.gather(loss_mean).mean().detach().item(), 
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }
            if args.use_proj_loss:
                logs["proj_loss"] = accelerator.gather(proj_loss_mean).mean().detach().item()
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--resume-step", type=int, default=0)

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of classes in the dataset.")

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--ema-decay", type=float, default=0.9999, help="EMA decay rate.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--proj-weight-schedule", type=str, default="hard", 
                        choices=["hard", "hard_high", "sigmoid", "cosine", "snr"],
                        help="Weighting schedule for projection loss. "
                             "'hard': binary threshold at tau (w=1 for t<tau); "
                             "'hard_high': inverse of hard (w=1 for t>=tau); "
                             "'sigmoid': smooth sigmoid transition centered at tau; "
                             "'cosine': cosine decay from 1 to 0 ending at tau; "
                             "'snr': SNR-based weighting with transition at tau.")
    parser.add_argument("--proj-tau", type=float, default=0.7, 
                        help="Split point (tau) for projection loss weighting. "
                             "For 'hard': cutoff where weight switches from 1 to 0. "
                             "For 'sigmoid': point where weight = 0.5. "
                             "For 'cosine': point where weight reaches 0. "
                             "For 'snr': point where weight = 0.5 based on SNR.")
    parser.add_argument("--proj-k", type=float, default=20.0,
                        help="Temperature/sharpness for sigmoid schedule. "
                             "Higher values = sharper transition (k~10: smooth, k~50: near-hard).")
    parser.add_argument("--weighting", default="uniform", type=str, help="Timestep sampling weighting.")
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--loss-type", type=str, default="si", choices=["si", "stablevm"], help="Loss type to use.")
    parser.add_argument("--use-proj-loss", action=argparse.BooleanOptionalAction, default=False, help="Whether to use projection loss.")
    parser.add_argument("--bank-capacity-per-class", type=int, default=256, help="Memory bank capacity per class for StableVMLoss.")
    parser.add_argument("--bank-on-gpu", action=argparse.BooleanOptionalAction, default=True, 
                        help="Store memory bank on GPU (faster, ~2GB VRAM) or CPU with pinned memory (slower, saves VRAM).")
    parser.add_argument("--prefill-bank-fully", action=argparse.BooleanOptionalAction, default=False, help="If set, prefill the memory bank to capacity per class before training; otherwise only ensure at least one per class.")
    parser.add_argument("--bank-chunk-size", type=int, default=1024, help="Chunk size when iterating over reference bank tensors to bound memory and traffic.")
    parser.add_argument("--prefill-batch-size", type=int, default=1024, help="Batch size for fast memory bank prefill when --prefill-bank-fully is set.")
    parser.add_argument("--prefill-workers", type=int, default=None, help="Worker count for the prefill dataloader (defaults to --num-workers if None).")
    
    # iREPA parameters
    parser.add_argument("--use-irepa", action=argparse.BooleanOptionalAction, default=False, 
                        help="Use iREPA improvements: Conv projector and spatial normalization on encoder features.")
    parser.add_argument("--irepa-gamma", type=float, default=0.6,
                        help="Gamma coefficient for iREPA spatial normalization: x = x - gamma * x.mean(dim=1).")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
