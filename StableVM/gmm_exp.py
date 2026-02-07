"""
Toy experiments of 2D distributions for stable vector field
We consider using CFM, STF and StableVM to model the distribution of a GMM.
We use the same model architecture for all three methods, and all experiments are conducted under the setting of linear interpolation.
Author: Donglin Yang
Date: 2025-11-14
"""
import os
import re
import math
import glob
import imageio
import argparse
from tqdm import tqdm
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from dataset import CustomDataset
from torch.utils.data import DataLoader

@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):    
    """Sample from VAE posterior distribution."""
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z

@torch.no_grad()
def generate_gmm_data(means, variances, pi, sample_size=100000, seed=None):
    """
    Fast GMM samples generation
    
    Args:
        means:      (n_components, num_features)
        variances:  (n_components, 1) or (n_components, num_features) or scalar
        pi:         (n_components,)
        sample_size: sample size (Default 100k)
        seed:       random seed (Default None for random results)
    
    Return:
        samples: (sample_size, 2)
    """   
    # Get device from input tensors
    device = means.device
    
    # Save current random state
    rng_state = torch.get_rng_state()
    if device.type == 'cuda':
        cuda_rng_state = torch.cuda.get_rng_state()
    
    # Only set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
    
    # sample index
    # use multinomial to generate indices
    component_indices = torch.multinomial(
        pi, 
        num_samples=sample_size, 
        replacement=True
    )  # shape: (sample_size,)
    
    # select means and variances
    selected_means = means[component_indices]      # shape: (sample_size, num_features)
    # if variance is scalar, just calculate the std
    if isinstance(variances, (int, float)):
        selected_stds = math.sqrt(variances)  # shape: (sample_size, num_features)
    else:
        variances = variances[component_indices]  # shape: (sample_size, num_features)
        selected_stds = torch.sqrt(variances)  # shape: (sample_size, num_features)
    
    # generate noise
    noise = torch.randn_like(selected_means)      # shape: (sample_size, num_features)
    samples = selected_means + noise * selected_stds
    
    # Restore random state
    torch.set_rng_state(rng_state)
    if device.type == 'cuda':
        torch.cuda.set_rng_state(cuda_rng_state)
    
    return samples

@torch.no_grad()
def perturb_data_gaussian(x, t, seed=None):
    """
    Perturb data with Gaussian noise, VP formulation
    Args:
        x: (n_samples, ...)
        t: noise level (n_samples,)
        seed: random seed (Default None for random results)
    Returns:
        perturbed samples: (n_samples, 2)
    """
    assert x.shape[0] == t.shape[0], f"x and t must have the same number of samples, but got {x.shape[0]} and {t.shape[0]}"
    # Save current random state
    rng_state = torch.get_rng_state()
    if x.device.type == 'cuda':
        cuda_rng_state = torch.cuda.get_rng_state()
    
    # Only set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if x.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)

    perturbed = (1-t.unsqueeze(1)) * x + torch.randn_like(x) * t.unsqueeze(1)
    
    # Restore random state
    torch.set_rng_state(rng_state)
    if x.device.type == 'cuda':
        torch.cuda.set_rng_state(cuda_rng_state)
    
    return perturbed

@torch.no_grad()
def perturb_data_gmm(x, t, seed=None):
    """
    Perturb data with gmm distribution
    Args:
        x: (n_reference_samples, 2)
        t: noise level (sample_size,)
        seed: random seed (Default None for random results)
    Returns:
        perturbed samples: (sample_size, 2)
    """
    # Save current random state
    rng_state = torch.get_rng_state()
    if x.device.type == 'cuda':
        cuda_rng_state = torch.cuda.get_rng_state()
    
    # Only set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if x.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)

    b = x.shape[0]
    sample_size = t.shape[0]
    indices = torch.randint(0, b, (sample_size,))
    
    # sample from GMM
    means = x[indices] * (1-t).unsqueeze(1)  # (sample_size, 2)
    perturb_data = torch.randn_like(means) * t.unsqueeze(1) + means  # (sample_size, 2)
    
    # Restore random state
    torch.set_rng_state(rng_state)
    if x.device.type == 'cuda':
        torch.cuda.set_rng_state(cuda_rng_state)
    
    return perturb_data

@torch.no_grad()
def cfm_target(x, b, t, vector_field="velocity"):
    """
    Compute conditional flow matching target field
    Parameters:
        x: (N, d) tensor
        b: (N, d) tensor (reference samples)
        t: noise parameter (N,)
        vector_field: "velocity" or "score"
    Returns:
        target: (N, d) tensor
    """
    assert x.shape[0] == b.shape[0], f"x and b must have the same number of samples, but got {x.shape[0]} and {b.shape[0]}"
    if vector_field == "velocity":
        # Add numerical stability for t near 0
        t_safe = torch.clamp(t.unsqueeze(1), min=1e-4)
        target = (x -  b) / t_safe 
    elif vector_field == "score":
        # Add numerical stability for t near 0
        t_safe = torch.clamp(t.unsqueeze(1), min=1e-4)
        target = ((1-t.unsqueeze(1)) * b - x)/t_safe
    return target

@torch.no_grad()
def stable_target(x, b, t, vector_field="velocity", chunk_size=100):
    """
    Compute stable target vector field.
    Parameters:
        x: (N, d) tensor
        b: (M, d) tensor (reference samples)
        t: noise parameter, scaler or tensor (N,)
        vector_field: "velocity" or "score"
        chunk_size: size of chunks for processing reference batch (default: 1000)
    Returns:
        target: (N, d) tensor
    """
    if isinstance(t, torch.Tensor):
        assert x.shape[0] == t.shape[0], f"t must be a tensor of shape ({x.shape[0]},)"
        assert x.device == t.device, f"t must be on the same device as x, but got {x.device} and {t.device}"
    N, d = x.shape
    M = b.shape[0]
    dim = d

    # First compute log probabilities for all reference samples
    log_probs = []
    for i in range(0, M, chunk_size):
        b_chunk = b[i:i+chunk_size]  # (chunk_size, d)
        
        # Compute pairwise squared distances using cdist
        if isinstance(t, torch.Tensor):
            # For tensor t, we need to compute distances for each t value
            # Scale reference samples for each t value
            b_chunk_scaled = b_chunk.unsqueeze(0) * (1-t.unsqueeze(1).unsqueeze(2))  # (N, chunk_size, d)
            # Compute squared distances
            dist = torch.sum((x.unsqueeze(1) - b_chunk_scaled) ** 2, dim=-1)  # (N, chunk_size)
        else:
            b_chunk_scaled = b_chunk * (1-t)  # (chunk_size, d)
            # Compute squared distances
            dist = torch.cdist(x, b_chunk_scaled, p=2)**2  # (N, chunk_size)
        
        # Add numerical stability for small t
        if isinstance(t, torch.Tensor):
            t_safe = torch.clamp(t.unsqueeze(1), min=1e-6)
        else:
            t_safe = max(t, 1e-6)
        
        # Use a more stable form for log probability calculation
        log_prob = -dist / (2 * t_safe**2)
        log_probs.append(log_prob)
    
    # Concatenate all log probabilities and compute weights
    log_probs = torch.cat(log_probs, dim=1)  # (N, M)
    
    # Add numerical stability for log_softmax
    weights = torch.softmax(log_probs, dim=1)  # (N, M)
    # clip small weights smaller than 1e-5 to zero
    weights = torch.where(weights < 1e-5, torch.zeros_like(weights), weights)


    # Initialize target
    target = torch.zeros_like(x)
    
    # Process reference batch in chunks
    for i in range(0, M, chunk_size):
        b_chunk = b[i:i+chunk_size]  # (chunk_size, d)
        weights_chunk = weights[:, i:i+chunk_size]  # (N, chunk_size)

        # Compute vector field efficiently
        if vector_field == "velocity":
            # For velocity field: v = \Sum_i w_i * (x - b_i) / t
            if isinstance(t, torch.Tensor):
                t_safe = torch.clamp(t.unsqueeze(1).unsqueeze(2), min=1e-6)  # (N, 1, 1)
            else:
                t_safe = max(t, 1e-6)
            b_exp = b_chunk.unsqueeze(0)  # (1, chunk_size, d)
            x_exp = x.unsqueeze(1)  # (N, 1, d)
            target_chunk = (x_exp - b_exp) / t_safe  # (N, chunk_size, d)
            target += (weights_chunk.unsqueeze(2) * target_chunk).sum(dim=1)  # (N, d)
        elif vector_field == "score":  # score field
            # For score field: s = -\Sum_i w_i * (x - (1-t)b_i)/t^2
            if isinstance(t, torch.Tensor):
                t_safe = torch.clamp(t.unsqueeze(1), min=1e-6)
                b_chunk_scaled = b_chunk.unsqueeze(0) * (1-t.unsqueeze(1).unsqueeze(2))  # (N, chunk_size, d)
            else:
                t_safe = max(t, 1e-6)
                b_chunk_scaled = b_chunk.unsqueeze(0) * (1-t)  # (1, chunk_size, d)
            x_expanded = x.unsqueeze(1)  # (N, 1, d)
            vec_diff = x_expanded - b_chunk_scaled  # (N, chunk_size, d)
            target_chunk = -(weights_chunk.unsqueeze(2) * vec_diff).sum(dim=1) / t_safe  # (N, d)

            target += target_chunk
        else:
            raise ValueError(f"Unsupported vector_field: {vector_field}")

    return target

@torch.no_grad()
def stable_target_efficient(x, b, t, vector_field="velocity", ref_chunk_size=1024, x_chunk_size=1024):
    """
    Memory-efficient stable target computation using streaming log-sum-exp.
    Supports very large reference sets by avoiding materializing (N, M) weights.

    Args:
        x: (N, d) tensor of perturbed samples
        b: (M, d) tensor of reference samples
        t: scalar float, or (N,) tensor of noise levels
        vector_field: "velocity" or "score"
        ref_chunk_size: size for chunking reference set
        x_chunk_size: size for chunking x set
    Returns:
        target: (N, d) tensor
    """
    device = x.device
    N, d = x.shape
    M = b.shape[0]

    # Handle t
    if isinstance(t, torch.Tensor):
        assert t.shape[0] == N, f"t tensor must have shape ({N},)"
        t_tensor = t.to(device)
    else:
        t_tensor = torch.full((N,), float(t), device=device)

    # Clamp for numerical stability
    t_safe_row = torch.clamp(t_tensor, min=1e-6)
    inv_two_t2 = 1.0 / (2.0 * (t_safe_row ** 2))  # shape (N,)

    # Output buffer
    target = torch.zeros_like(x)

    # Process x in chunks
    for xs in range(0, N, x_chunk_size):
        xe = min(xs + x_chunk_size, N)
        x_chunk = x[xs:xe]  # (Bx, d)
        Bx = x_chunk.shape[0]
        t_row = t_safe_row[xs:xe]  # (Bx,)
        inv_two_t2_row = inv_two_t2[xs:xe]  # (Bx,)

        # First pass: compute row-wise max logits for numerical stability
        row_max = torch.full((Bx,), -float('inf'), device=device)
        for rs in range(0, M, ref_chunk_size):
            re = min(rs + ref_chunk_size, M)
            b_chunk = b[rs:re].to(device)  # (Br, d)

            # Compute logits = -||x - (1-t)b||^2 / (2 t^2)
            b_scaled = b_chunk.unsqueeze(0) * (1.0 - t_row.view(Bx, 1, 1))  # (Bx, Br, d)
            x_exp = x_chunk.unsqueeze(1)  # (Bx, 1, d)
            dist2 = torch.sum((x_exp - b_scaled) ** 2, dim=-1)  # (Bx, Br)
            logits = -dist2 * inv_two_t2_row.view(Bx, 1)  # (Bx, Br)
            row_max = torch.maximum(row_max, logits.max(dim=1).values)

        # Second pass: accumulate denom and numerator
        denom = torch.zeros((Bx,), device=device)
        num = torch.zeros((Bx, d), device=device)
        for rs in range(0, M, ref_chunk_size):
            re = min(rs + ref_chunk_size, M)
            b_chunk = b[rs:re].to(device)  # (Br, d)

            b_scaled = b_chunk.unsqueeze(0) * (1.0 - t_row.view(Bx, 1, 1))  # (Bx, Br, d)
            x_exp = x_chunk.unsqueeze(1)  # (Bx, 1, d)
            dist2 = torch.sum((x_exp - b_scaled) ** 2, dim=-1)  # (Bx, Br)
            logits = -dist2 * inv_two_t2_row.view(Bx, 1)  # (Bx, Br)
            weights_unnorm = torch.exp(logits - row_max.view(Bx, 1))  # (Bx, Br)
            denom += weights_unnorm.sum(dim=1)  # (Bx,)

            if vector_field == "velocity":
                vec = (x_exp - b_chunk.unsqueeze(0)) / t_row.view(Bx, 1, 1)  # (Bx, Br, d)
                num += (weights_unnorm.unsqueeze(-1) * vec).sum(dim=1)
            elif vector_field == "score":
                vec = (x_exp - b_scaled) / t_row.view(Bx, 1, 1)  # (Bx, Br, d)
                num += (weights_unnorm.unsqueeze(-1) * (-vec)).sum(dim=1)
            else:
                raise ValueError(f"Unsupported vector_field: {vector_field}")

        # Normalize
        weights_norm = torch.clamp(denom, min=1e-12)
        target[xs:xe] = num / weights_norm.view(Bx, 1)

    return target

def visualize_model_vs_true_field(model, reference_batch, device, t, epoch, out_dir="compare_vecfield", grid_size=20, xlim=(-5, 5), ylim=(-5, 5), field_type="score"):
    os.makedirs(out_dir, exist_ok=True)

    x = torch.linspace(xlim[0], xlim[1], grid_size)
    y = torch.linspace(ylim[0], ylim[1], grid_size)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device)

    with torch.no_grad():
        t = torch.tensor(t, device=device, dtype=torch.float32).repeat(grid.shape[0])
        pred_scores = model(grid, t).cpu().numpy()
        true_scores = stable_target(grid, reference_batch, t, vector_field=field_type).cpu().numpy()

    up = pred_scores[:, 0].reshape(grid_size, grid_size)
    vp = pred_scores[:, 1].reshape(grid_size, grid_size)
    ut = true_scores[:, 0].reshape(grid_size, grid_size)
    vt = true_scores[:, 1].reshape(grid_size, grid_size)

    plt.figure(figsize=(10, 10))
    
    # Plot both fields on the same plot
    plt.quiver(xx, yy, up, vp, color='blue', angles='xy', alpha=0.6, label='Model Field')
    plt.quiver(xx, yy, ut, vt, color='red', angles='xy', alpha=0.6, label='True Field')
    
    plt.title(f"{field_type.capitalize()} Fields Comparison (Epoch {epoch})")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    plt.savefig(os.path.join(out_dir, f"compare_vecfield_{epoch:05d}.png"))
    plt.close()

def create_gif_from_frames(frame_dir="compare_vecfield_frames", gif_path="compare_vector_field.gif", duration=0.3):
    frame_paths = sorted(glob.glob(os.path.join(frame_dir, "compare_vecfield_*.png")))

    # filter index ≤ 10000 
    filtered_paths = []
    for path in frame_paths:
        match = re.search(r"compare_vecfield_(\d+).png", os.path.basename(path))
        if match and int(match.group(1)) <= 10000:
            filtered_paths.append(path)

    # save image
    images = [imageio.imread(p) for p in filtered_paths]
    imageio.mimsave(gif_path, images, duration=duration)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        # Increase the base frequency to make different t values more distinguishable
        self.base_freq = 1000  # Increased from 10000 to make frequencies higher

    def forward(self, t):
        if t.ndim == 1:
            t = t[:, None]  # shape (B, 1)
        half_dim = self.embed_dim // 2
        # Scale t to [0, 2π] to make better use of the sinusoidal range
        t_scaled = t * 2 * math.pi
        freqs = torch.exp(
            -math.log(self.base_freq) * torch.arange(half_dim, dtype=torch.float32, device=t.device) / half_dim
        )  # (half_dim,)
        args = t_scaled * freqs  # (B, half_dim)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, embed_dim)
        return embedding

class AdaLNZero(nn.Module):
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(cond_dim, hidden_dim * 2)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, cond):
        scale_shift = self.fc(cond)  # (B, 2 * hidden_dim)
        scale, shift = scale_shift.chunk(2, dim=-1)
        x_norm = self.norm(x)
        return x_norm * (1 + scale) + shift

class ScoreNet(nn.Module):
    def __init__(self, x_dim=2, hidden_dim=32, time_embed_dim=32):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.input_layer = nn.Linear(x_dim + time_embed_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                AdaLNZero(hidden_dim, time_embed_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(4)
        ])
        self.output_layer = nn.Linear(hidden_dim, x_dim)

    def forward(self, x, t):
        t_embed = self.time_embed(t)  # (B, time_embed_dim)
        h = torch.cat([x, t_embed], dim=1)
        h = self.input_layer(h)
        for block in self.blocks:
            h = block[1](block[0](h, t_embed))  # AdaLNZero + Linear
            h = block[2](h)
            h = block[3](h)
        return self.output_layer(h)

class EMA:
    def __init__(self, beta):
        self.beta = beta
        self.shadow = {}
    
    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.beta) * param.data + self.beta * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_ema(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

def plot_variance_curve(means, variances, pi, device, vector_field="score", num_timesteps=50, num_samples=10000, seed=42):
    """
    Compute variance curves for CFM and STF at different timesteps.
    
    Args:
        means: GMM means
        variances: GMM variances
        pi: GMM mixing weights
        device: torch device
        vector_field: "velocity" or "score" (default: "score")
        num_timesteps: number of timesteps to evaluate (default: 50)
        num_samples: number of samples to use for evaluation (default: 10000)
        seed: random seed (default: 42)
    
    Returns:
        timesteps_np: numpy array of timesteps
        divergences_mean: mean divergence for each timestep (unnormalized)
        divergences_q15: 15th percentile across samples for each timestep
        divergences_q85: 85th percentile across samples for each timestep
    """
    # Get data dimensionality
    d = means.shape[1]
    
    # Generate reference samples
    reference_samples = generate_gmm_data(means, variances, pi, sample_size=num_samples, seed=seed)
    
    # Generate evaluation samples
    eval_samples = reference_samples
    
    # Create timesteps
    timesteps = torch.linspace(0.01, 0.99, num_timesteps, device=device)
    
    divergences_mean = torch.zeros(num_timesteps, device=device)
    divergences_q15 = torch.zeros(num_timesteps, device=device)
    divergences_q85 = torch.zeros(num_timesteps, device=device)
    # Compute divergences at each timestep
    for i, t in tqdm(enumerate(timesteps), desc="Computing divergences"):
        # Perturb samples
        t_tensor = t.repeat(num_samples)
        perturbed_samples = perturb_data_gaussian(eval_samples, t_tensor, seed=seed+2)
        
        # Compute CFM target
        cfm_targets = cfm_target(perturbed_samples, eval_samples, t_tensor, vector_field=vector_field)
        
        # Compute STF target using stable_target
        stf_targets = stable_target(perturbed_samples, reference_samples, t_tensor, vector_field=vector_field)
        
        # Compute Fisher divergence per sample, then aggregate mean and quantiles
        per_sample = ((cfm_targets - stf_targets) ** 2).sum(dim=1)
        divergences_mean[i] = per_sample.mean()
        divergences_q15[i] = torch.quantile(per_sample, 0.15)
        divergences_q85[i] = torch.quantile(per_sample, 0.85)
    
    return (
        timesteps.cpu().numpy(),
        divergences_mean.cpu().numpy(),
        divergences_q15.cpu().numpy(),
        divergences_q85.cpu().numpy(),
    )

@torch.no_grad()
def plot_variance_curve_dataset(reference_samples, device, vector_field="score", num_timesteps=50, seed=42, x_chunk_size=512, ref_chunk_size=2048):
    """
    Compute variance curve for an arbitrary dataset using all samples as both eval and reference (in chunks).

    Args:
        reference_samples: (M, d) tensor of dataset samples
        device: torch device
        vector_field: "velocity" or "score"
        num_timesteps: number of timesteps
        seed: random seed
        x_chunk_size: chunk size for eval samples
        ref_chunk_size: chunk size for reference samples
    Returns:
        timesteps_np, mean, q15, q85
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    reference_samples = reference_samples.to(device)
    M, d = reference_samples.shape
    timesteps = torch.linspace(0.01, 0.99, num_timesteps, device=device)

    means = []
    q15s = []
    q85s = []

    for t in tqdm(timesteps, desc="Computing dataset divergences"):
        per_sample_list = []
        # Process eval samples in chunks to avoid O(M^2) memory
        for xs in range(0, M, x_chunk_size):
            xe = min(xs + x_chunk_size, M)
            eval_chunk = reference_samples[xs:xe]
            Bx = eval_chunk.shape[0]
            t_chunk = t.repeat(Bx)
            # Perturb
            perturbed_chunk = perturb_data_gaussian(eval_chunk, t_chunk, seed=seed+2)
            # CFM target against clean eval_chunk
            cfm_chunk = cfm_target(perturbed_chunk, eval_chunk, t_chunk, vector_field=vector_field)
            # STF target against full reference set using stable_target (chunked over reference)
            stf_chunk = stable_target(perturbed_chunk, reference_samples, t_chunk, vector_field=vector_field, chunk_size=ref_chunk_size)
            # Per-sample divergence
            per_sample = ((cfm_chunk - stf_chunk) ** 2).sum(dim=1)
            per_sample_list.append(per_sample)

        per_sample_all = torch.cat(per_sample_list, dim=0)
        means.append(per_sample_all.mean().item())
        q15s.append(torch.quantile(per_sample_all, 0.15).item())
        q85s.append(torch.quantile(per_sample_all, 0.85).item())

    return timesteps.cpu().numpy(), np.array(means), np.array(q15s), np.array(q85s)

def main(exp_type, vector_field, use_wandb=False, plot_variance=True, only_plot=False, enable_visualization=False, imagenet_data_path=None):
    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Create tensorboard writer
    # experiment parameters
    batch_size = 256
    if exp_type == "stf" or exp_type == "stable_vm" or exp_type == "stable_vm_efficient":
        reference_batch_size = 2048
    else:
        reference_batch_size = None
    
    # Create experiment name with reference_batch_size
    exp_name = f"{exp_type}_{vector_field}"
    if reference_batch_size is not None:
        exp_name = f"{exp_name}_ref{reference_batch_size}"
    
    log_dir = f"runs/{exp_name}"
    writer = SummaryWriter(log_dir=log_dir)

    ema_beta = 0.999
    lr = 1e-4
    num_updates = 20000  # Total number of model updates
    test_epoch = 500
    test_sample_size = 10000
    test_reference_sample_size = 100000
    visualize_t = 0.5
    visualize_epoch = 100

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_t_list = [0.2, 0.3, 0.4, 0.5]

    # Log experiment configuration
    config = {
        "exp_type": exp_type,
        "vector_field": vector_field,
        "batch_size": batch_size,
        "reference_batch_size": reference_batch_size,
        "ema_beta": ema_beta,
        "learning_rate": lr,
        "num_updates": num_updates,
        "test_epoch": test_epoch,
        "test_sample_size": test_sample_size,
        "test_reference_sample_size": test_reference_sample_size,
        "visualize_t": visualize_t,
        "visualize_epoch": visualize_epoch,
        "seed": seed,
        "test_t": test_t_list,
        "plot_variance": plot_variance,
        "only_plot": only_plot,
        "enable_visualization": enable_visualization
    }
    
    # Initialize wandb if enabled and not only plotting
    if use_wandb and not only_plot:
        wandb.init(
            project="toy-exp",
            name=exp_name,
            config=config
        )
    
    # Log config to tensorboard
    for key, value in config.items():
        writer.add_text(f"config/{key}", str(value))

    # Plot variance curve for different dimensions
    if plot_variance:
        logger.info(f"Plotting variance curves for different dimensions...")
        dims = [10, 100, 500]
        n_components = 100
        num_samples = 10000
        
        # Set style (consistent with plot_variance_from_npz.py)
        plt.style.use('seaborn-v0_8-paper')
        
        # Create figure with specific size and DPI for publication
        plt.figure(figsize=(8, 4.5), dpi=300)
        
        # Define a professional color palette (consistent with plot_variance_from_npz.py)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        # Ensure data directory exists for saving plot data
        os.makedirs('plots/data', exist_ok=True)
        
        # Plot each dimension with different line styles
        for idx, dim in enumerate(dims):
            npz_path = f'plots/data/variance_{vector_field}_dim{dim}.npz'
            
            # Check if saved data exists, otherwise compute
            if os.path.exists(npz_path):
                logger.info(f"Loading existing data for dim={dim} from {npz_path}")
                data = np.load(npz_path)
                timesteps_np = data["t"]
                divergences_mean = data["mean"]
                divergences_q15 = data["q15"]
                divergences_q85 = data["q85"]
            else:
                logger.info(f"Computing variance curve for dim={dim}...")
                # Generate GMM parameters for current dimension
                means = np.random.uniform(-1, 1, size=(n_components, dim))
                variances = np.random.uniform(1e-2, 1e-1, size=(n_components, dim))
                pi = np.random.uniform(0.1, 1.0, size=n_components)
                pi /= np.sum(pi)
                
                # Convert to tensors
                means = torch.tensor(means, dtype=torch.float32, device=device)
                variances = torch.tensor(variances, dtype=torch.float32, device=device)
                pi = torch.tensor(pi, dtype=torch.float32, device=device)
                
                # Plot variance curve for current dimension
                timesteps_np, divergences_mean, divergences_q15, divergences_q85 = plot_variance_curve(
                    means, variances, pi, device, vector_field=vector_field,
                    num_timesteps=50, num_samples=num_samples, seed=seed
                )
                
                # Save curve data for this dimension
                np.savez(
                    npz_path,
                    t=timesteps_np,
                    mean=divergences_mean,
                    q15=divergences_q15,
                    q85=divergences_q85,
                    dim=np.array(dim)
                )
                logger.info(f"Saved variance data to {npz_path}")
            
            # Normalize by sqrt(dim)
            scale = np.sqrt(dim)
            mean_n = divergences_mean / scale
            q15_n = divergences_q15 / scale
            q85_n = divergences_q85 / scale
            
            # Plot mean curve with shaded variance region (mean ± std)
            plt.plot(timesteps_np, mean_n, 
                    color=colors[idx],
                    label=f'GMM (dim={dim})',
                    linewidth=2,
                    marker='o',
                    markersize=7,
                    markevery=5)  # Add marker every 5 points
            lower_band = np.maximum(q15_n, 0)
            upper_band = q85_n
            plt.fill_between(
                timesteps_np,
                lower_band,
                upper_band,
                color=colors[idx],
                alpha=0.2
            )

        # Add CIFAR-10 curve using all 50k training samples
        cifar_npz_path = f'plots/data/variance_{vector_field}_cifar10.npz'
        
        if os.path.exists(cifar_npz_path):
            logger.info(f"Loading existing CIFAR-10 data from {cifar_npz_path}")
            data = np.load(cifar_npz_path)
            t_np = data["t"]
            mean_np = data["mean"]
            q15_np = data["q15"]
            q85_np = data["q85"]
        else:
            logger.info("Computing CIFAR-10 variance curve...")
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            # Stack all 50k samples to a tensor of shape (50000, 32*32*3)
            cifar_data = torch.stack([img.view(-1) for img, _ in cifar_train], dim=0).to(device)
            # Normalize to roughly match synthetic scale
            cifar_data = (cifar_data - 0.5) / 0.5
            # Plot CIFAR variance curve
            t_np, mean_np, q15_np, q85_np = plot_variance_curve_dataset(
                cifar_data, device, vector_field=vector_field, num_timesteps=20, seed=seed,
                x_chunk_size=256, ref_chunk_size=1024
            )
            # Save CIFAR-10 curve data
            np.savez(
                cifar_npz_path,
                t=t_np,
                mean=mean_np,
                q15=q15_np,
                q85=q85_np,
                num_samples=np.array(cifar_data.shape[0])
            )
            logger.info(f"Saved CIFAR-10 variance data to {cifar_npz_path}")
        
        # Normalize by sqrt(dim), CIFAR-10 dimensionality = 32*32*3 = 3072
        cifar_scale = np.sqrt(32*32*3)
        mean_n_cifar = mean_np / cifar_scale
        q15_n_cifar = q15_np / cifar_scale
        q85_n_cifar = q85_np / cifar_scale
        
        plt.plot(t_np, mean_n_cifar,
                 color='#d62728',
                 label='CIFAR-10',
                 linewidth=2,
                 marker='s',
                 markersize=7,
                 markevery=2)  # More markers since only 20 timesteps
        plt.fill_between(
            t_np,
            np.maximum(q15_n_cifar, 0),
            q85_n_cifar,
            color='#d62728',
            alpha=0.15
        )

        # Add ImageNet curve
        imagenet_npz_path = f'plots/data/variance_{vector_field}_imagenet.npz'
        imagenet_loaded = False
        
        # First try to load from saved NPZ file
        if os.path.exists(imagenet_npz_path):
            logger.info(f"Loading existing ImageNet data from {imagenet_npz_path}")
            data = np.load(imagenet_npz_path)
            t_np_imagenet = data["t"]
            mean_np_imagenet = data["mean"]
            q15_np_imagenet = data["q15"]
            q85_np_imagenet = data["q85"]
            imagenet_loaded = True
        elif imagenet_data_path is not None and os.path.exists(imagenet_data_path):
            # Compute from raw data if NPZ doesn't exist
            logger.info(f"Loading ImageNet dataset from {imagenet_data_path}...")
            try:
                imagenet_dataset = CustomDataset(imagenet_data_path)
                imagenet_dataloader = DataLoader(
                    imagenet_dataset,
                    batch_size=100,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True
                )
                
                # Collect 50k ImageNet latents
                logger.info("Collecting 50k ImageNet latents...")
                imagenet_latents_list = []
                imagenet_count = 0
                
                latents_scale = torch.tensor([0.18215, 0.18215, 0.18215, 0.18215]).view(1, 4, 1, 1).to(device)
                latents_bias = -torch.tensor([0., 0., 0., 0.]).view(1, 4, 1, 1).to(device)
                
                for batch in imagenet_dataloader:
                    raw_images, vae_features, labels = batch
                    vae_features = vae_features.to(device)
                    # Sample from posterior and apply scaling/bias
                    latents = sample_posterior(vae_features, latents_scale=latents_scale, latents_bias=latents_bias)
                    # Flatten latents to (batch_size, 4*32*32)
                    latents_flat = latents.view(latents.size(0), -1)
                    imagenet_latents_list.append(latents_flat)
                    
                    imagenet_count += latents.size(0)
                    if imagenet_count >= 50000:
                        break
                
                imagenet_latents = torch.cat(imagenet_latents_list, dim=0)[:50000]
                logger.info(f"Collected {imagenet_latents.shape[0]} ImageNet latents with shape {imagenet_latents.shape}")
                
                # Plot ImageNet variance curve
                t_np_imagenet, mean_np_imagenet, q15_np_imagenet, q85_np_imagenet = plot_variance_curve_dataset(
                    imagenet_latents, device, vector_field=vector_field, num_timesteps=20, seed=seed,
                    x_chunk_size=256, ref_chunk_size=1024
                )
                
                # Save ImageNet curve data
                np.savez(
                    imagenet_npz_path,
                    t=t_np_imagenet,
                    mean=mean_np_imagenet,
                    q15=q15_np_imagenet,
                    q85=q85_np_imagenet,
                    num_samples=np.array(imagenet_latents.shape[0])
                )
                logger.info(f"Saved ImageNet variance data to {imagenet_npz_path}")
                imagenet_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load ImageNet dataset: {e}")
                logger.warning("Skipping ImageNet variance curve...")
        elif imagenet_data_path is not None:
            logger.warning(f"ImageNet data path not found: {imagenet_data_path}")
            logger.warning("Skipping ImageNet variance curve. Please check the path.")
        else:
            logger.info("No saved ImageNet data found and no data path specified. Skipping ImageNet curve.")
        
        # Plot ImageNet curve if data was loaded
        if imagenet_loaded:
            # Normalize by sqrt(dim), ImageNet latents are 32*32*4 = 4096
            imagenet_scale = np.sqrt(32*32*4)
            mean_n_imagenet = mean_np_imagenet / imagenet_scale
            q15_n_imagenet = q15_np_imagenet / imagenet_scale
            q85_n_imagenet = q85_np_imagenet / imagenet_scale
            
            plt.plot(t_np_imagenet, mean_n_imagenet,
                     color='#9467bd',  # Purple color
                     label='ImageNet latents',
                     linewidth=2,
                     marker='^',
                     markersize=7,
                     markevery=1)  # More markers since only 20 timesteps
            plt.fill_between(
                t_np_imagenet,
                np.maximum(q15_n_imagenet, 0),
                q85_n_imagenet,
                color='#9467bd',
                alpha=0.15
            )
        
        # Customize the plot (consistent with plot_variance_from_npz.py)
        plt.xlabel('Timestep $t$', fontsize=12, fontweight='bold')
        plt.ylabel(r'Variance / $\sqrt{d}$', fontsize=12, fontweight='bold')
        plt.title('Variance Curves', fontsize=14, fontweight='bold', pad=15)
        
        # Set axis limits with symlog scale for better visualization of small values
        plt.xlim(0, 1)
        plt.yscale('symlog', linthresh=1, linscale=0.5)  # Linear near 0, log elsewhere
        plt.ylim(0, 60)
        plt.yticks([0, 10, 20, 30, 40, 50, 60])
        
        # Customize legend (consistent with plot_variance_from_npz.py)
        plt.legend(fontsize=12, frameon=True, framealpha=0.95, 
                  edgecolor='black', fancybox=False)
        
        # Adjust layout and add tight padding
        plt.tight_layout()
        
        # Save with high quality
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/variance_curves_{vector_field}_combined.png', 
                   dpi=300, 
                   bbox_inches='tight',
                   pad_inches=0.1)
        plt.close()
    
    # If only_plot is True, exit after plotting
    if only_plot:
        logger.info("Only plotting variance curves, exiting...")
        writer.close()
        return

    # GMM distribution for training
    n_components = 100
    dim = 10  # Use 2D for training
    means = np.random.uniform(-1, 1, size=(n_components, dim))
    variances = np.random.uniform(1e-2, 1e-1, size=(n_components, dim))
    pi = np.random.uniform(0.1, 1.0, size=n_components)
    pi /= np.sum(pi)

    means = torch.tensor(means, dtype=torch.float32, device=device)
    variances = torch.tensor(variances, dtype=torch.float32, device=device)
    pi = torch.tensor(pi, dtype=torch.float32, device=device)
    
    # generate test samples
    test_samples = generate_gmm_data(means, variances, pi, sample_size=test_sample_size, seed=seed)
    # generate reference samples
    test_reference_samples = generate_gmm_data(means, variances, pi, sample_size=test_reference_sample_size, seed=seed)

    model = ScoreNet(x_dim=dim).to(device)
    model.train()
    ema_model = ScoreNet(x_dim=dim).to(device)
    ema_model.eval()
    ema = EMA(beta=ema_beta)
    ema.register(model)
    ema.apply_ema(ema_model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    progress_bar = tqdm(range(num_updates), desc=f"Training {exp_name}", unit="update")

    for update in progress_bar:
        model.train()
        if exp_type == "cfm":
            x_batch = generate_gmm_data(means, variances, pi, sample_size=batch_size, seed=seed+update)
            t = torch.rand((batch_size,), device=device)
            # Add small epsilon to avoid numerical instability
            t = torch.clamp(t, min=1e-5, max=1.0)
            x_batch_perturbed = perturb_data_gaussian(x_batch, t, seed=seed+update+1)
            target = cfm_target(x_batch_perturbed, x_batch, t, vector_field=vector_field)
            pred = model(x_batch_perturbed, t)
            loss = torch.mean(torch.sum((pred - target) ** 2, dim=1))

            optimizer.zero_grad()
            loss.backward()
            # Add gradient clipping for CFM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)
        elif exp_type == "stf" or exp_type == "stable_vm" or exp_type == "stable_vm_efficient":
            
            reference_batch = generate_gmm_data(means, variances, pi, sample_size=reference_batch_size, seed=seed+update)
            
            if exp_type == "stf":
                t = torch.rand(batch_size, device=device)  # Uniform[0, 1] noise levels for training
                x_batch_perturbed = reference_batch[0].repeat(batch_size, 1) * (1 - t.view(-1, 1)) + torch.randn((batch_size, dim), device=device) * t.view(-1, 1)
                x_batch_perturbed = perturb_data_gaussian(x_batch, t, seed=seed+update+1)
                target = stable_target(x_batch_perturbed, reference_batch, t, vector_field=vector_field)
            elif exp_type == "stable_vm":
                t = torch.rand(batch_size, device=device)  # Uniform[0, 1] noise levels for training
                x_batch_perturbed = perturb_data_gmm(reference_batch, t, seed=seed+update+1)
                target = stable_target(x_batch_perturbed, reference_batch, t, vector_field=vector_field)
            else:  # stable_vm_efficient
                t = torch.rand(batch_size, device=device)  # Uniform[0, 1] noise levels for training
                x_batch_perturbed = perturb_data_gmm(reference_batch, t, seed=seed+update+1)
                target = stable_target_efficient(x_batch_perturbed, reference_batch, t, vector_field=vector_field, sigma_threshold=3.0, chunk_size=1000)
            
            pred = model(x_batch_perturbed, t)
            loss = torch.mean(torch.sum((pred - target) ** 2, dim=1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)
        else:
            raise ValueError(f"Unsupported exp_type: {exp_type}")

        progress_bar.set_postfix({"Update": f"{update + 1}/{num_updates}", "Loss": f"{loss.item():.4f}"})
        writer.add_scalar("loss", loss.item(), update+1)
        
        # Log to wandb if enabled
        if use_wandb:
            wandb.log({
                "loss": loss.item(),
                "update": update + 1
            })

        if (update + 1) % test_epoch == 0 or update == 0:
            ema.apply_ema(ema_model)
            fisher_div_avg = 0.0
            # Collect per-t stats for saving
            per_t_means = []
            for t_val in test_t_list:
                t = torch.tensor(t_val, device=device, dtype=torch.float32).repeat(test_sample_size)
                test_samples_perturbed = perturb_data_gaussian(test_samples, t, seed=seed+update)
                pred_scores = ema_model(test_samples_perturbed, t)
                true_scores = stable_target(test_samples_perturbed, test_reference_samples, t, vector_field=vector_field)
                per_sample = torch.sum((pred_scores - true_scores) ** 2, dim=1)
                fisher_div = per_sample.mean()
                logger.info(f"Update [{update + 1}/{num_updates}], t={t_val:.2f}, Fisher Divergence: {fisher_div.item():.4f}")
                writer.add_scalar(f"Fisher_Divergence/t_{t_val:.2f}", fisher_div.item(), update+1)
                if use_wandb:
                    wandb.log({
                        f"fisher_divergence/t_{t_val:.2f}": fisher_div.item(),
                        "update": update + 1
                    })
                fisher_div_avg += fisher_div.item()
                per_t_means.append(fisher_div.item())
                
            fisher_div_avg /= len(test_t_list)
            writer.add_scalar("Fisher_Divergence/avg", fisher_div_avg, update+1)
            if use_wandb:
                wandb.log({
                    "fisher_divergence/avg": fisher_div_avg,
                    "update": update + 1
                })
            # Save per-evaluation-step stats for plotting
            os.makedirs('plots/data/eval', exist_ok=True)
            np.savez(
                f"plots/data/eval/eval_step_{update + 1:06d}_{vector_field}.npz",
                step=np.array(update + 1),
                t=np.array(test_t_list, dtype=np.float32),
                mean=np.array(per_t_means, dtype=np.float32),
                exp_name=np.array(exp_name)
            )
            # Also save into method-specific compare folder for cross-run comparison (CFM vs StableVM)
            method_name = 'cfm' if exp_type == 'cfm' else ('stable_vm' if exp_type in ['stf', 'stable_vm', 'stable_vm_efficient'] else exp_type)
            compare_dir = os.path.join('plots', 'data', 'compare', method_name)
            os.makedirs(compare_dir, exist_ok=True)
            np.savez(
                os.path.join(compare_dir, f"fisher_step_{update + 1:06d}_{vector_field}.npz"),
                step=np.array(update + 1),
                t=np.array(test_t_list, dtype=np.float32),
                mean=np.array(per_t_means, dtype=np.float32),
                exp_name=np.array(exp_name),
                method=np.array(method_name)
            )
                
        if enable_visualization and (((update + 1) % visualize_epoch == 0 and (update+1) <= 10000) or update == 0):
            # visualize vector fields
            visualize_model_vs_true_field(ema_model, test_reference_samples, test_reference_samples.device, visualize_t, update+1, 
                                        out_dir=f"compare_vecfield_{exp_name}", field_type=vector_field)

    # create GIF from frames (optional)
    if enable_visualization:
        gif_path = f"compare_field_{exp_name}.gif"
        create_gif_from_frames(f"compare_vecfield_{exp_name}", gif_path)
        
        # Upload GIF to wandb if enabled
        if use_wandb:
            wandb.log({
                "vector_field_gif": wandb.Video(gif_path),
                "update": update + 1
            })
    
    # Close the writer
    writer.close()
    
    # Close wandb if enabled
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp", choices=["cfm","stf", "stable_vm", "stable_vm_efficient"], type=str, default="stf",
        help="Choose which experiment to run"
    )
    parser.add_argument(
        "--vector_field", choices=["velocity", "score"], type=str, default="velocity",
        help="Choose which vector field to use"
    )
    parser.add_argument(
        "--use_wandb", action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--no_plot_variance", action="store_true",
        help="Disable variance curve plotting"
    )
    parser.add_argument(
        "--only_plot", action="store_true",
        help="Only plot variance curve without training"
    )
    parser.add_argument(
        "--enable_visualization", action="store_true",
        help="Enable vector field visualization"
    )
    parser.add_argument(
        "--imagenet-data-path", type=str, default=None,
        help="Path to ImageNet dataset for variance curve evaluation (optional)"
    )
    args = parser.parse_args()
    main(args.exp, args.vector_field, args.use_wandb, not args.no_plot_variance, args.only_plot, args.enable_visualization, args.imagenet_data_path)