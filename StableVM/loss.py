import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


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

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

# expand the tensor to the same dimensionality of another tensor (N,) to (N, 1, ..., 1)
def expand_tensor(x, shape):
    return x.view(x.shape[0], *((1,) * (len(shape) - 1)))


#################################################################################
#                     Projection Loss Weighting Schedules                       #
#################################################################################

def proj_weight_hard(t: torch.Tensor, tau: float = 0.7, **kwargs) -> torch.Tensor:
    """
    Hard threshold weighting: w(t) = 1 if t < tau, else 0.
    
    This is a binary mask that completely enables alignment for t < tau
    and completely disables it for t >= tau.
    
    Args:
        t: Time values (N,) in [0, 1], where 0 = data, 1 = noise
        tau: Split point (default: 0.7)
    
    Returns:
        weights: (N,) tensor of weights in {0, 1}
    """
    return (t < tau).float()


def proj_weight_hard_high(t: torch.Tensor, tau: float = 0.7, **kwargs) -> torch.Tensor:
    """
    Hard threshold weighting for high noise region: w(t) = 1 if t >= tau, else 0.
    
    This is the inverse of 'hard' - enables alignment only for t >= tau (high noise region).
    Used as an ablation to compare with the standard 'hard' schedule.
    
    Args:
        t: Time values (N,) in [0, 1], where 0 = data, 1 = noise
        tau: Split point (default: 0.7)
    
    Returns:
        weights: (N,) tensor of weights in {0, 1}
    """
    return (t >= tau).float()


def proj_weight_sigmoid(t: torch.Tensor, tau: float = 0.7, k: float = 20.0, **kwargs) -> torch.Tensor:
    """
    Parametric sigmoid weighting: w(t) = sigmoid(k * (tau - t))
    
    This provides a smooth, differentiable transition from 1 to 0 centered at tau.
    At t = tau, the weight is exactly 0.5.
    
    Args:
        t: Time values (N,) in [0, 1], where 0 = data, 1 = noise
        tau: Split point where weight = 0.5 (default: 0.7)
        k: Temperature/sharpness parameter (default: 20.0)
           - k ≈ 10: Smooth transition (soft)
           - k ≈ 50: Near-vertical drop (approximates hard threshold)
           - k → ∞: Converges to hard threshold
    
    Returns:
        weights: (N,) tensor of weights in (0, 1)
    """
    return torch.sigmoid(k * (tau - t))


def proj_weight_cosine(t: torch.Tensor, tau: float = 0.85, **kwargs) -> torch.Tensor:
    """
    Cosine weighting with variable cutoff: smooth decay from 1 to 0.
    
    w(t) = 0.5 * (1 + cos(pi * t / tau_max))  if t < tau_max
         = 0                                   if t >= tau_max
    
    This guarantees weights are bounded exactly within [0, 1] and provides
    a natural, smooth decay. The weight equals 1 at t=0 and 0 at t=tau.
    
    Args:
        t: Time values (N,) in [0, 1], where 0 = data, 1 = noise
        tau: Maximum time where weight becomes 0 (default: 0.85)
             The weight will be naturally high in [0, 0.5*tau] and decay to 0 by tau.
    
    Returns:
        weights: (N,) tensor of weights in [0, 1]
    """
    # Compute cosine weight for t < tau
    weight = 0.5 * (1.0 + torch.cos(np.pi * t / tau))
    # Zero out weight for t >= tau
    weight = torch.where(t < tau, weight, torch.zeros_like(weight))
    return weight


def proj_weight_snr(t: torch.Tensor, tau: float = 0.7, path_type: str = "linear", **kwargs) -> torch.Tensor:
    """
    SNR-based weighting: w(t) = SNR(t) / (SNR(t) + SNR_limit)
    
    Since the posterior width is strictly a function of the Signal-to-Noise Ratio,
    this weighting is theoretically motivated and robust to noise scheduler changes.
    
    For linear path (z_t = (1-t)*x + t*eps):
        SNR(t) = (1-t)^2 / t^2
    
    For cosine path:
        SNR(t) = cos^2(t*pi/2) / sin^2(t*pi/2)
    
    The weight is:
    - ~1 when SNR(t) >> SNR_limit (high signal, near data)
    - ~0 when SNR(t) << SNR_limit (low signal, near noise)  
    - 0.5 when SNR(t) = SNR_limit (at the split point)
    
    Args:
        t: Time values (N,) in [0, 1], where 0 = data, 1 = noise
        tau: Split point (default: 0.7). SNR_limit is computed from this value.
        path_type: "linear" or "cosine" interpolation path
    
    Returns:
        weights: (N,) tensor of weights in (0, 1)
    """
    # Clamp t to avoid division by zero at boundaries
    t_clamped = t.clamp(min=1e-6, max=1.0 - 1e-6)
    tau_clamped = max(min(tau, 1.0 - 1e-6), 1e-6)
    
    if path_type == "linear":
        # SNR(t) = (1-t)^2 / t^2
        snr_t = ((1.0 - t_clamped) / t_clamped) ** 2
        snr_limit = ((1.0 - tau_clamped) / tau_clamped) ** 2
    elif path_type == "cosine":
        # SNR(t) = cos^2(t*pi/2) / sin^2(t*pi/2) = cot^2(t*pi/2)
        angle_t = t_clamped * np.pi / 2
        angle_tau = tau_clamped * np.pi / 2
        snr_t = (torch.cos(angle_t) / torch.sin(angle_t)) ** 2
        snr_limit = (np.cos(angle_tau) / np.sin(angle_tau)) ** 2
    else:
        raise ValueError(f"Unknown path_type: {path_type}")
    
    # w(t) = SNR(t) / (SNR(t) + SNR_limit)
    weight = snr_t / (snr_t + snr_limit)
    return weight


# Registry of available projection weight schedules
PROJ_WEIGHT_SCHEDULES = {
    "hard": proj_weight_hard,
    "hard_high": proj_weight_hard_high,
    "sigmoid": proj_weight_sigmoid, 
    "cosine": proj_weight_cosine,
    "snr": proj_weight_snr,
}


def get_proj_weight_fn(schedule: str):
    """
    Get a projection weight function by name.
    
    Args:
        schedule: One of "hard", "sigmoid", "cosine", "snr"
    
    Returns:
        Callable weight function with signature (t, tau, **kwargs) -> weights
    """
    if schedule not in PROJ_WEIGHT_SCHEDULES:
        raise ValueError(
            f"Unknown projection weight schedule: {schedule}. "
            f"Available: {list(PROJ_WEIGHT_SCHEDULES.keys())}"
        )
    return PROJ_WEIGHT_SCHEDULES[schedule]

class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            use_proj_loss=True,
            proj_weight_schedule="hard",
            proj_tau=0.7,
            proj_k=20.0,
            ):
        """
        Args:
            prediction: Prediction target type ('v' for velocity)
            path_type: Interpolation path type ("linear" or "cosine")
            weighting: Timestep sampling weighting ("uniform" or "lognormal")
            encoders: List of encoders for projection loss
            accelerator: Accelerator for distributed training
            latents_scale: Scale factor for VAE latents
            latents_bias: Bias for VAE latents
            use_proj_loss: Whether to use projection loss
            proj_weight_schedule: Weighting schedule for projection loss.
                - "hard": Binary threshold, w(t) = 1 if t < tau, else 0
                - "sigmoid": Smooth sigmoid, w(t) = sigmoid(k * (tau - t))
                - "cosine": Cosine decay, w(t) = 0.5 * (1 + cos(pi * t / tau)) for t < tau
                - "snr": SNR-based, w(t) = SNR(t) / (SNR(t) + SNR(tau))
            proj_tau: Split point / threshold for projection loss weighting (default: 0.7)
                - For "hard": Cutoff point where weight switches from 1 to 0
                - For "sigmoid": Point where weight = 0.5
                - For "cosine": Point where weight reaches 0
                - For "snr": Point where weight = 0.5 (based on SNR at this time)
            proj_k: Temperature/sharpness for sigmoid schedule (default: 20.0)
                - Only used with "sigmoid" schedule
                - Higher values → sharper transition (more like hard threshold)
                - k ≈ 10: smooth, k ≈ 50: near-vertical
        """
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        self.use_proj_loss = use_proj_loss
        self.proj_weight_schedule = proj_weight_schedule
        self.proj_tau = proj_tau
        self.proj_k = proj_k
        self.proj_weight_fn = get_proj_weight_fn(proj_weight_schedule)

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1 * torch.ones_like(t)
            d_sigma_t =  torch.ones_like(t)
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, model_kwargs=None, zs=None):
        if model_kwargs == None:
            model_kwargs = {}
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            """
            sample timestep according to log-normal distribution of sigmas following EDM and follow the implementation in flow matching
            https://github.com/facebookresearch/flow_matching/blob/main/examples/image/training/train_loop.py#L26
            """
            P_mean = -1.2
            P_std = 1.2
            rnd_normal = torch.randn((images.shape[0], 1, 1, 1))
            sigma = (rnd_normal * P_std + P_mean).exp()
            time_input = 1 / (1 + sigma)
            time_input = torch.clip(time_input, min=0.0, max=1.0-1e-5)
            time_input = 1 - time_input
        else:
            raise NotImplementedError()
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        elif self.prediction == 'x':
            model_target = images
        else:
            raise NotImplementedError() # TODO: add x or eps prediction
        if self.use_proj_loss:
            model_output, zs_tilde  = model(model_input, time_input.flatten(), **model_kwargs)
        else:
            model_output = model(model_input, time_input.flatten(), **model_kwargs)
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        # projection loss with soft/hard weighting based on timestep
        if self.use_proj_loss and zs is not None and zs_tilde is not None:
            # Compute per-sample weights using the selected schedule
            t_flat = time_input.flatten()  # (N,)
            proj_weights = self.proj_weight_fn(
                t_flat, 
                tau=self.proj_tau, 
                k=self.proj_k,
                path_type=self.path_type
            )  # (N,)
            
            # Compute weighted sum of valid weights for normalization
            weight_sum = proj_weights.sum()
            
            if weight_sum > 1e-8:
                # Compute per-sample projection losses
                batch_size = t_flat.shape[0]
                num_encoders = len(zs)
                per_sample_proj_loss = torch.zeros(batch_size, device=images.device, dtype=images.dtype)
                
                for z, z_tilde in zip(zs, zs_tilde):
                    # z, z_tilde: (N, num_patches, embed_dim)
                    z_norm = F.normalize(z, dim=-1)
                    z_tilde_norm = F.normalize(z_tilde, dim=-1)
                    # Cosine similarity loss: -sum(z * z_tilde) per patch, averaged over patches
                    cosine_sim = (z_norm * z_tilde_norm).sum(dim=-1)  # (N, num_patches)
                    per_sample_proj_loss += - cosine_sim.mean(dim=-1)  # (N,)
                
                per_sample_proj_loss /= num_encoders  # Average over encoders
                
                # Apply soft weights and compute weighted average
                weighted_proj_loss = (proj_weights * per_sample_proj_loss).sum() / weight_sum
                proj_loss = weighted_proj_loss
            else:
                # All weights are ~0 (all samples are near noise), return zero tensor
                proj_loss = torch.zeros(1, device=images.device, dtype=images.dtype)
        else:
            proj_loss = torch.tensor(0., device=images.device, dtype=images.dtype)
        
        if self.use_proj_loss:
            return denoising_loss, proj_loss
        else:
            return denoising_loss

    def evaluate_loss_at_time(self, model, images, t, model_kwargs=None, zs=None):
        """
        Evaluate the loss for a batch of samples at a given time t.
        
        Args:
            model: The model to evaluate
            images: Input images batch (B, C, H, W)
            t: Time value (scalar)
            model_kwargs: Additional model arguments
            zs: Optional projection features
        
        Returns:
            Average loss at time t. If use_proj_loss is True, returns (total_loss, denoising_loss, proj_loss),
            otherwise returns denoising_loss.
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        # Ensure t is the right shape for broadcasting
        time_input = torch.full((images.shape[0], 1, 1, 1), t)
        
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        elif self.prediction == 'x':
            model_target = images
        else:
            raise NotImplementedError() # TODO: add x or eps prediction
        
        model_output = model(model_input, time_input.flatten(), **model_kwargs)
        
        # Handle tuple output when model has projections
        if isinstance(model_output, tuple):
            model_output = model_output[0]  # Take the main output, ignore projections
        
        denoising_loss = mean_flat((model_output - model_target) ** 2)
        
        return denoising_loss.mean()

    @torch.no_grad()
    def stable_target(self, model_input, ref_images, t, chunk_size=1000):
        """
        Compute stable target vector field.
        Parameters:
            model_input: (N, C, H, W) tensor
            ref_images: (M, C, H, W) tensor
            t: noise parameter, tensor (N,)
            chunk_size: size of chunks for processing reference batch (default: 1000)
        Returns:
            target: (N, C, H, W) tensor
        """
        M = ref_images.shape[0]
        t = t.clamp(min=1e-5)
        t_expanded = expand_tensor(t, model_input.shape) # (N, 1, 1, 1)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t_expanded) # (N, 1, 1, 1)

        # First compute log probabilities for all reference samples
        log_probs = []
        for i in range(0, M, chunk_size):
            b_chunk = ref_images[i:i+chunk_size]
            use_nb = (
                model_input.device.type == 'cuda'
                and b_chunk.device.type == 'cpu'
                and hasattr(b_chunk, 'is_pinned')
                and b_chunk.is_pinned()
            )
            # Convert to fp32 for computation while transferring to GPU
            b_chunk = b_chunk.to(device=model_input.device, dtype=torch.float32, non_blocking=use_nb)  # (chunk_size, C, H, W)

            # Compute pairwise squared distances
            b_chunk_scaled = b_chunk.unsqueeze(0) * alpha_t.unsqueeze(1)  # (N, chunk_size, C, H, W)
            # Compute squared distances
            dist = torch.sum((model_input.unsqueeze(1) - b_chunk_scaled) ** 2, dim=(2, 3, 4))  # (N, chunk_size)

            # Use a more stable form for log probability calculation
            log_prob = -dist / (2 * sigma_t.squeeze(-1).squeeze(-1)**2)
            log_probs.append(log_prob)

        # Concatenate all log probabilities and compute weights
        log_probs = torch.cat(log_probs, dim=1)  # (N, M)
        log_probs = log_probs - torch.max(log_probs, dim=1, keepdim=True).values
        # Add numerical stability for log_softmax
        weights = torch.softmax(log_probs, dim=1)  # (N, M)

        # Initialize target
        target = torch.zeros_like(model_input)

        # Process reference batch in chunks
        for i in range(0, M, chunk_size):
            b_chunk = ref_images[i:i+chunk_size]
            use_nb = (
                model_input.device.type == 'cuda'
                and b_chunk.device.type == 'cpu'
                and hasattr(b_chunk, 'is_pinned')
                and b_chunk.is_pinned()
            )
            # Convert to fp32 for computation while transferring to GPU
            b_chunk = b_chunk.to(device=model_input.device, dtype=torch.float32, non_blocking=use_nb)  # (chunk_size, C, H, W)
            weights_chunk = weights[:, i:i+chunk_size]  # (N, chunk_size)
            weights_chunk = weights_chunk.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (N, chunk_size, 1, 1, 1)
            
            # Compute vector field efficiently
            if self.prediction == 'v':
                b_exp = b_chunk.unsqueeze(0)  # (1, chunk_size, C, H, W)
                alpha_t_exp = alpha_t.unsqueeze(1)  # (N, 1, 1, 1, 1)
                sigma_t_exp = sigma_t.unsqueeze(1)  # (N, 1, 1, 1, 1)
                d_alpha_t_exp = d_alpha_t.unsqueeze(1)  # (N, 1, 1, 1, 1)
                d_sigma_t_exp = d_sigma_t.unsqueeze(1)  # (N, 1, 1, 1, 1)
            
                d_sigma_t_sigma_t_ratio = d_sigma_t_exp / sigma_t_exp
                b_scale = (d_alpha_t_exp - d_sigma_t_sigma_t_ratio * alpha_t_exp) * b_exp # (N, chunk_size, C, H, W)
                x_exp = model_input.unsqueeze(1)  # (N, 1, C, H, W)
                x_scale = d_sigma_t_sigma_t_ratio * x_exp # (N, 1, C, H, W)
                target_chunk = x_scale + b_scale # (N, chunk_size, C, H, W)
            elif self.prediction == 'x':
                target_chunk = b_chunk.unsqueeze(0)
            else:
                raise NotImplementedError()
            target += (weights_chunk * target_chunk).sum(dim=1)  # (N, C, H, W)

        return target

class StableVMLoss:
    def __init__(self, batch_size=256, num_classes=1000, prediction='v',
                 path_type="linear", weighting="uniform", cfg_prob=0.1, bank_chunk_size: int = 1024,
                 use_proj_loss: bool = False, proj_weight_schedule: str = "hard",
                 proj_tau: float = 0.7, proj_k: float = 20.0,
                 encoders=None, encoder_types=None, preprocess_fn=None, dataset=None,
                 use_irepa: bool = False, irepa_gamma: float = 1.0):
        """
        Args:
            batch_size: Batch size for training
            num_classes: Number of classes in the dataset
            prediction: Prediction target type ('v' for velocity, 'x' for data)
            path_type: Interpolation path type ("linear" or "cosine")
            weighting: Timestep sampling weighting ("uniform" or "lognormal")
            cfg_prob: Classifier-free guidance dropout probability
            bank_chunk_size: Chunk size for memory bank operations
            use_proj_loss: Whether to use projection/feature alignment loss
            proj_weight_schedule: Weighting schedule for projection loss.
                - "hard": Binary threshold, w(t) = 1 if t < tau, else 0
                - "sigmoid": Smooth sigmoid, w(t) = sigmoid(k * (tau - t))
                - "cosine": Cosine decay, w(t) = 0.5 * (1 + cos(pi * t / tau)) for t < tau
                - "snr": SNR-based, w(t) = SNR(t) / (SNR(t) + SNR(tau))
            proj_tau: Split point / threshold for projection loss weighting (default: 0.7)
            proj_k: Temperature/sharpness for sigmoid schedule (default: 20.0)
            encoders: List of encoder models for computing features on-the-fly
            encoder_types: List of encoder type strings (e.g., 'dinov2', 'clip')
            preprocess_fn: Function to preprocess raw images for encoders
            dataset: Dataset object for loading raw images by index (required for proj_loss)
            use_irepa: Whether to use iREPA spatial normalization on encoder features
            irepa_gamma: Gamma coefficient for iREPA spatial normalization (default: 1.0)
        """
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.cfg_prob = cfg_prob
        # Efficiency knobs
        self.bank_chunk_size = bank_chunk_size
        # Projection loss parameters
        self.use_proj_loss = use_proj_loss
        self.proj_weight_schedule = proj_weight_schedule
        self.proj_tau = proj_tau
        self.proj_k = proj_k
        # Encoders for on-the-fly feature computation
        self.encoders = encoders
        self.encoder_types = encoder_types
        self.preprocess_fn = preprocess_fn
        self.dataset = dataset
        # iREPA parameters
        self.use_irepa = use_irepa
        self.irepa_gamma = irepa_gamma
        if use_proj_loss:
            self.proj_weight_fn = get_proj_weight_fn(proj_weight_schedule)
            if encoders is None:
                raise ValueError("encoders must be provided when use_proj_loss=True")
            if dataset is None:
                raise ValueError("dataset must be provided when use_proj_loss=True for on-the-fly image loading")

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -torch.ones_like(t)
            d_sigma_t = torch.ones_like(t)
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t = np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    @torch.no_grad()
    def _stable_target(self, perturbed_images, ref_images, t, chunk_size=None):
        """
        chunked vectorized version: avoid O(N*M) loop + control memory usage
        Optimized for GPU-resident memory bank (no CPU->GPU transfer overhead)
        """
        if chunk_size is None:
            chunk_size = self.bank_chunk_size
        N = perturbed_images.shape[0]
        
        # Numerical stability: clamp t to avoid division by zero and edge cases
        t = t.clamp(min=1e-4, max=1.0 - 1e-4)

        t_exp = t.view(N, 1, 1, 1)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t_exp)
        # Force fp32 for all computations to avoid numerical issues
        alpha_t = alpha_t.to(dtype=torch.float32)
        sigma_t = sigma_t.to(dtype=torch.float32)
        d_alpha_t = d_alpha_t.to(dtype=torch.float32)
        d_sigma_t = d_sigma_t.to(dtype=torch.float32)
        perturbed_images = perturbed_images.to(dtype=torch.float32)
        
        # Check if ref_images is already on the same device (GPU-resident bank)
        ref_on_same_device = ref_images.device == perturbed_images.device

        # initialize log_probs storage
        log_probs_chunks = []

        for start in range(0, ref_images.shape[0], chunk_size):
            ref_chunk = ref_images[start:start+chunk_size]
            if ref_on_same_device:
                # Already on GPU, just convert dtype
                ref_chunk = ref_chunk.to(dtype=torch.float32)
            else:
                # CPU -> GPU transfer (legacy path)
                use_nb = (
                    perturbed_images.device.type == 'cuda'
                    and ref_chunk.device.type == 'cpu'
                    and hasattr(ref_chunk, 'is_pinned')
                    and ref_chunk.is_pinned()
                )
                ref_chunk = ref_chunk.to(device=perturbed_images.device, dtype=torch.float32, non_blocking=use_nb)

            ref_scaled = ref_chunk.unsqueeze(0) * alpha_t.unsqueeze(1)  # (N, chunk, C, H, W)
            diff = perturbed_images.unsqueeze(1) - ref_scaled
            dist = diff.pow(2).sum(dim=(2, 3, 4))  # (N, chunk)

            # Add epsilon to sigma_t to prevent division by zero
            sigma_t_sq = (sigma_t.view(N, -1)[:, 0] ** 2 + 1e-8).unsqueeze(1)
            log_prob = -dist / (2 * sigma_t_sq)
            # Clamp log_prob to prevent extreme values
            log_prob = log_prob.clamp(min=-1e4)
            log_probs_chunks.append(log_prob)

        log_probs = torch.cat(log_probs_chunks, dim=1)
        # Numerical stability: subtract max for log-sum-exp trick
        log_probs_max = log_probs.max(dim=1, keepdim=True).values
        log_probs = log_probs - log_probs_max
        # Clamp to prevent underflow in exp
        log_probs = log_probs.clamp(min=-50)
        weights = torch.softmax(log_probs, dim=1)  # (N, M)
        
        # Check for NaN in weights and replace with 0
        nan_mask = torch.isnan(weights).any(dim=1)
        if nan_mask.any():
            M = weights.shape[1]
            weights[nan_mask] = 0

        # compute target
        target = torch.zeros_like(perturbed_images)

        for start in range(0, ref_images.shape[0], chunk_size):
            ref_chunk = ref_images[start:start+chunk_size]
            if ref_on_same_device:
                # Already on GPU, just convert dtype
                ref_chunk = ref_chunk.to(dtype=torch.float32)
            else:
                # CPU -> GPU transfer (legacy path)
                use_nb = (
                    perturbed_images.device.type == 'cuda'
                    and ref_chunk.device.type == 'cpu'
                    and hasattr(ref_chunk, 'is_pinned')
                    and ref_chunk.is_pinned()
                )
                ref_chunk = ref_chunk.to(device=perturbed_images.device, dtype=torch.float32, non_blocking=use_nb)
            w_chunk = weights[:, start:start+chunk_size]  # (N, chunk)
            w_chunk = w_chunk.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (N, chunk, 1, 1, 1)

            if self.prediction == 'v':
                alpha_t_exp = alpha_t.unsqueeze(1)
                sigma_t_exp = sigma_t.unsqueeze(1)
                d_alpha_t_exp = d_alpha_t.unsqueeze(1)
                d_sigma_t_exp = d_sigma_t.unsqueeze(1)

                # Add epsilon to sigma_t to prevent division by zero
                d_sigma_ratio = d_sigma_t_exp / (sigma_t_exp + 1e-8)
                b_scale = (d_alpha_t_exp - d_sigma_ratio * alpha_t_exp) * ref_chunk.unsqueeze(0)
                x_scale = d_sigma_ratio * perturbed_images.unsqueeze(1)
                target_chunk = x_scale + b_scale
            elif self.prediction == 'x':
                target_chunk = ref_chunk.unsqueeze(0)
            else:
                raise NotImplementedError()

            target += (w_chunk * target_chunk).sum(dim=1)

        return target

    @torch.no_grad()
    def _stable_target_from_bank(self, perturbed_images: torch.Tensor, t: torch.Tensor, bank, input_labels: torch.Tensor, chunk_size: Optional[int] = None):
        """
        Compute stable target using only the memory bank corresponding to each sample's label
        """
        target = torch.zeros_like(perturbed_images, dtype=torch.float32)
        # Track which samples have been processed (for fallback)
        processed = torch.zeros(perturbed_images.shape[0], dtype=torch.bool, device=perturbed_images.device)
        
        if chunk_size is None:
            chunk_size = self.bank_chunk_size

        labels = input_labels.unique()
        for lbl in labels.tolist():
            cls = int(lbl)
            idx = input_labels == cls
            ref = bank.data[cls]
            if ref.shape[0] == 0:
                # Try to use the global pool (index num_classes) as fallback
                ref = bank.data[bank.num_classes]
                if ref.shape[0] == 0:
                    continue
            target[idx] = self._stable_target(
                perturbed_images[idx], ref, t[idx], chunk_size=chunk_size
            )
            processed[idx] = True
        
        # For any unprocessed samples, the target remains zero which will be handled
        # by the fallback in __call__
        
        return target

    def __call__(self, model, memory_bank, device):
        """
        StableVM training objective using a per-class memory bank.
        pick samples from the corresponding bank with the same label for perturbation,
          and compute stable target using only that bank.
        
        Returns:
            If use_proj_loss is False: denoising_loss (N,)
            If use_proj_loss is True: (denoising_loss, proj_loss) tuple
        """
        # Conditional branch with memory bank
        assert memory_bank is not None, "memory_bank is required for conditional StableVM"
        
        # determine device and batch size to sample
        device = device if device is not None else memory_bank.data[0].device
        batch_size = self.batch_size

        # sample labels and apply CFG dropout
        input_labels = torch.randint(0, self.num_classes, (batch_size,), device=device)
        if self.cfg_prob > 0:
            dropout_mask = torch.rand(batch_size, device=device) < self.cfg_prob
            input_labels[dropout_mask] = self.num_classes

        # sample t with proper clamping for numerical stability
        if self.weighting == "uniform":
            t = torch.rand((batch_size,), device=device)
            # Clamp to avoid edge cases at t=0 and t=1
            t = t.clamp(min=1e-4, max=1.0 - 1e-4)
        elif self.weighting == "lognormal":
            P_mean, P_std = -1.2, 1.2
            rnd_normal = torch.randn((batch_size, 1, 1, 1), device=device)
            sigma = (rnd_normal * P_std + P_mean).exp()
            t = 1 / (1 + sigma)
            t = torch.clip(t, min=1e-4, max=1.0 - 1e-4)
            t = 1 - t.squeeze()
        else:
            raise NotImplementedError()
        
        # Sample from memory bank, optionally with dataset indices for on-the-fly feature computation
        sel_imgs, dataset_indices = memory_bank.sample_by_labels(
            input_labels, 
            return_indices=self.use_proj_loss
        )
        
        # Ensure fp32 for all computations
        sel_imgs = sel_imgs.to(dtype=torch.float32)

        t_exp = t.view(-1, 1, 1, 1)
        alpha_t, sigma_t, _, _ = self.interpolant(t_exp)
        alpha_t = alpha_t.to(dtype=torch.float32)
        sigma_t = sigma_t.to(dtype=torch.float32)
        noises = torch.randn_like(sel_imgs)
        perturbed_images = alpha_t * sel_imgs + sigma_t * noises

        # time split
        target = torch.zeros_like(perturbed_images)
        idx_simple = (t < 0.5)
        idx_complex = ~idx_simple

        if idx_simple.any():
            t_simple = t[idx_simple].view(-1, 1, 1, 1)
            _, _, d_alpha_t, d_sigma_t = self.interpolant(t_simple)
            d_alpha_t = d_alpha_t.to(dtype=torch.float32)
            d_sigma_t = d_sigma_t.to(dtype=torch.float32)
            noises_simple = noises[idx_simple]
            sel_imgs_simple = sel_imgs[idx_simple]
            if self.prediction == 'v':
                target[idx_simple] = d_alpha_t * sel_imgs_simple + d_sigma_t * noises_simple
            elif self.prediction == 'x':
                target[idx_simple] = sel_imgs_simple
            else:
                raise NotImplementedError()

        if idx_complex.any():    
            target[idx_complex] = self._stable_target_from_bank(
                perturbed_images[idx_complex], t[idx_complex], memory_bank, input_labels[idx_complex]
            )

        # Final safety check: replace any NaN/Inf in target with simple FM target
        nan_inf_mask = torch.isnan(target) | torch.isinf(target)
        if nan_inf_mask.any():
            # Fallback to simple flow matching target for problematic samples
            t_fallback = t.view(-1, 1, 1, 1)
            _, _, d_alpha_t_fb, d_sigma_t_fb = self.interpolant(t_fallback)
            d_alpha_t_fb = d_alpha_t_fb.to(dtype=torch.float32)
            d_sigma_t_fb = d_sigma_t_fb.to(dtype=torch.float32)
            fallback_target = d_alpha_t_fb * sel_imgs + d_sigma_t_fb * noises
            target = torch.where(nan_inf_mask, fallback_target, target)

        model_kwargs = dict(y=input_labels)
        
        # Forward pass - handle projection output if using projection loss
        if self.use_proj_loss:
            model_output, zs_tilde = model(perturbed_images, t, **model_kwargs)
        else:
            model_output = model(perturbed_images, t, **model_kwargs)
            zs_tilde = None
        
        denoising_loss = mean_flat((model_output - target) ** 2)

        # Compute projection loss with on-the-fly encoder feature computation
        if self.use_proj_loss and dataset_indices is not None and zs_tilde is not None:
            # Load raw images from dataset using indices (parallel loading for speed)
            valid_mask = dataset_indices >= 0
            if valid_mask.any():
                valid_indices = dataset_indices[valid_mask].cpu().tolist()
                
                # Parallel loading using ThreadPoolExecutor for faster IO
                def load_raw_image(idx):
                    return self.dataset[idx][0]  # First element is raw image
                
                with ThreadPoolExecutor(max_workers=8) as executor:
                    raw_images_list = list(executor.map(load_raw_image, valid_indices))
                
                # Stack into batch tensor
                raw_images = torch.stack(raw_images_list, dim=0).to(device=device, dtype=torch.float32)
                
                # Compute encoder features on-the-fly from raw images
                zs = []
                with torch.no_grad():
                    for encoder, encoder_type in zip(self.encoders, self.encoder_types):
                        # Preprocess raw images
                        if self.preprocess_fn is not None:
                            raw_processed = self.preprocess_fn(raw_images, encoder_type)
                        else:
                            raw_processed = raw_images / 255.0  # Default: just scale to [0, 1]
                        
                        z = encoder.forward_features(raw_processed)
                        if 'mocov3' in encoder_type:
                            z = z[:, 1:]
                        if 'dinov2' in encoder_type:
                            z = z['x_norm_patchtokens']
                        # Apply iREPA spatial normalization if enabled
                        if self.use_irepa:
                            z = irepa_spatial_normalize(z, gamma=self.irepa_gamma)
                        zs.append(z)
            else:
                zs = []
            
            if len(zs) > 0 and len(zs) == len(zs_tilde):
                # Compute per-sample weights using the selected schedule (only for valid samples)
                t_valid = t[valid_mask]
                proj_weights = self.proj_weight_fn(
                    t_valid, 
                    tau=self.proj_tau, 
                    k=self.proj_k,
                    path_type=self.path_type
                )  # (num_valid,)
                
                # Compute weighted sum of valid weights for normalization
                weight_sum = proj_weights.sum()
                
                if weight_sum > 1e-8:
                    # Compute per-sample projection losses (only for valid samples)
                    num_encoders = len(zs)
                    num_valid = valid_mask.sum().item()
                    per_sample_proj_loss = torch.zeros(num_valid, device=device, dtype=torch.float32)
                    
                    # Get model projections only for valid samples
                    zs_tilde_valid = [z_tilde[valid_mask] for z_tilde in zs_tilde]
                    
                    for z, z_tilde in zip(zs, zs_tilde_valid):
                        # z, z_tilde: (num_valid, num_patches, embed_dim)
                        z_norm = F.normalize(z, dim=-1)
                        z_tilde_norm = F.normalize(z_tilde, dim=-1)
                        # Cosine similarity loss: -sum(z * z_tilde) per patch, averaged over patches
                        cosine_sim = (z_norm * z_tilde_norm).sum(dim=-1)  # (num_valid, num_patches)
                        per_sample_proj_loss += -cosine_sim.mean(dim=-1)  # (num_valid,)
                    
                    per_sample_proj_loss /= num_encoders  # Average over encoders
                    
                    # Apply soft weights and compute weighted average
                    weighted_proj_loss = (proj_weights * per_sample_proj_loss).sum() / weight_sum
                    proj_loss = weighted_proj_loss
                else:
                    # All weights are ~0 (all samples are near noise), return zero tensor
                    proj_loss = torch.zeros(1, device=device, dtype=torch.float32)
            else:
                # No valid encoder features or mismatch
                proj_loss = torch.zeros(1, device=device, dtype=torch.float32)
        else:
            proj_loss = None

        if self.use_proj_loss:
            return denoising_loss, proj_loss
        else:
            return denoising_loss


class MemoryBank:
    """Per-class memory bank for latent features/images with optional dataset index storage."""
    def __init__(self, num_classes: int, capacity_per_class: int, feature_shape, device='cpu', dtype=torch.float16, pin_memory: bool = True, store_indices: bool = False):
        self.num_classes = num_classes
        self.capacity_per_class = capacity_per_class
        self.feature_shape = feature_shape
        self.device = device
        self.dtype = dtype  # Storage dtype (fp16)
        self.compute_dtype = torch.float32  # Computation dtype (fp32)
        self.store_indices = store_indices

        # Determine if we're on GPU
        self.on_gpu = device is not None and str(device) != 'cpu' and 'cuda' in str(device)
        # pin_memory only makes sense for CPU tensors
        self.pin_memory = pin_memory and not self.on_gpu

        # Allocate per-class banks
        if self.on_gpu:
            # Store on GPU for faster access during training
            self.data = [torch.empty((0, *feature_shape), device=device, dtype=dtype) for _ in range(num_classes + 1)]
        else:
            # Store on CPU (optionally pinned) to save GPU memory
            self.data = [
                torch.empty((0, *feature_shape), dtype=dtype, pin_memory=self.pin_memory)
                for _ in range(num_classes + 1)
            ]
        
        # Dataset index storage for on-the-fly raw image loading (always on CPU)
        # This is extremely memory efficient: just store int64 indices (~8 bytes per sample)
        # Total for ImageNet with 256 samples/class: 1001 * 256 * 8 bytes ≈ 2 MB
        if store_indices:
            self.indices = [
                torch.empty((0,), dtype=torch.int64)
                for _ in range(num_classes + 1)
            ]
        else:
            self.indices = None

    @torch.no_grad()
    def add(self, features: torch.Tensor, labels: torch.Tensor, dataset_indices: Optional[torch.Tensor] = None):
        """Add features to per-class banks with FIFO replacement when exceeding capacity.
        features: (N, C, H, W), labels: (N,)
        dataset_indices: Optional dataset indices (N,) for on-the-fly raw image loading
        """
        assert features.ndim == 4, "features should be (N,C,H,W)"
        assert labels.ndim == 1 and labels.shape[0] == features.shape[0]
        if dataset_indices is not None and self.store_indices:
            assert dataset_indices.ndim == 1, "dataset_indices should be (N,)"
            assert dataset_indices.shape[0] == features.shape[0], "dataset_indices size must match features"
        
        # Move to backing device/dtype for storage
        features = features.detach().to(self.device, dtype=self.dtype, non_blocking=True)
        # Keep indices on CPU as int64 (always CPU for dataset indexing)
        if dataset_indices is not None and self.store_indices:
            dataset_indices = dataset_indices.detach().to('cpu', dtype=torch.int64)
        
        labels_cpu = labels.detach().to('cpu')
        # For GPU storage, we need idx on the same device for index_select
        if self.on_gpu:
            idx_device = self.device
        else:
            idx_device = 'cpu'
        
        for cls in labels_cpu.unique().tolist():
            cls = int(cls)
            idx = (labels_cpu == cls).nonzero(as_tuple=True)[0].to(idx_device)
            feats_cls = features.index_select(0, idx)
            if feats_cls.numel() == 0:
                continue
            bank = self.data[cls]
            concat = torch.cat([bank, feats_cls], dim=0)
            if concat.shape[0] > self.capacity_per_class:
                # keep the most recent capacity_per_class entries (FIFO)
                concat = concat[-self.capacity_per_class:]
            # Preserve pinned memory when applicable (only for CPU tensors)
            if not self.on_gpu and self.pin_memory:
                concat = concat.pin_memory()
            self.data[cls] = concat
            
            # Also store dataset indices if provided (always on CPU)
            if dataset_indices is not None and self.indices is not None:
                idx_cpu = (labels_cpu == cls).nonzero(as_tuple=True)[0]
                idx_cls = dataset_indices.index_select(0, idx_cpu)
                idx_bank = self.indices[cls]
                idx_concat = torch.cat([idx_bank, idx_cls], dim=0)
                if idx_concat.shape[0] > self.capacity_per_class:
                    idx_concat = idx_concat[-self.capacity_per_class:]
                self.indices[cls] = idx_concat
        
        # Maintain an unlabeled bank (index == num_classes) as a global pool
        bank_no_label = self.data[self.num_classes]
        concat = torch.cat([bank_no_label, features], dim=0)
        if concat.shape[0] > self.capacity_per_class:
            concat = concat[-self.capacity_per_class:]
        if not self.on_gpu and self.pin_memory:
            concat = concat.pin_memory()
        self.data[self.num_classes] = concat
        
        # Also store indices for the global pool
        if dataset_indices is not None and self.indices is not None:
            idx_bank_no_label = self.indices[self.num_classes]
            idx_concat = torch.cat([idx_bank_no_label, dataset_indices], dim=0)
            if idx_concat.shape[0] > self.capacity_per_class:
                idx_concat = idx_concat[-self.capacity_per_class:]
            self.indices[self.num_classes] = idx_concat

    def size_for_class(self, cls: int) -> int:
        return 0 if cls < 0 or cls >= self.num_classes else self.data[cls].shape[0]

    @torch.no_grad()
    def sample_by_labels(self, labels: torch.Tensor, device: Optional[torch.device] = None, return_indices: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample one feature per requested label.
        For label == num_classes, pick a random available class.
        Returns:
            samples: (N, C, H, W) in compute_dtype (fp32) on target device
            dataset_indices: Optional (N,) dataset indices if return_indices=True and stored
        """
        device = labels.device if device is None else device
    
        # Sample from memory bank and convert to compute dtype
        samples = torch.empty((len(labels),) + self.feature_shape, device=device, dtype=self.compute_dtype)
        
        # Initialize index storage if needed
        if return_indices and self.indices is not None:
            dataset_indices = torch.full((len(labels),), -1, dtype=torch.int64, device=device)
            rand_indices_per_label = {}  # Store random indices for consistent sampling
        else:
            dataset_indices = None
            rand_indices_per_label = None
        
        unique_labels = labels.unique()
        for lbl in unique_labels:
            lbl_int = int(lbl)
            idx_mask = (labels == lbl).nonzero(as_tuple=True)[0]
            bank = self.data[lbl_int]
            if bank.shape[0] == 0 or len(idx_mask) == 0:
                samples[idx_mask] = 0
                continue
            # Generate random indices on the same device as the bank for efficient indexing
            rand_idx = torch.randint(bank.shape[0], (len(idx_mask),), device=bank.device)
            if rand_indices_per_label is not None:
                rand_indices_per_label[lbl_int] = (idx_mask, rand_idx.cpu())  # Store CPU version for indices
            # When bank is on GPU, this is just a dtype conversion (no device transfer)
            sampled_data = bank[rand_idx].to(dtype=self.compute_dtype)
            if sampled_data.device != device:
                sampled_data = sampled_data.to(device=device, non_blocking=True)
            samples[idx_mask] = sampled_data
        
        # Sample dataset indices using the same random indices (indices are always on CPU)
        if return_indices and self.indices is not None and rand_indices_per_label is not None:
            for lbl in unique_labels:
                lbl_int = int(lbl)
                if lbl_int not in rand_indices_per_label:
                    continue
                idx_mask, rand_idx = rand_indices_per_label[lbl_int]
                idx_bank = self.indices[lbl_int]
                
                if idx_bank.shape[0] == 0:
                    continue
                
                # Sample with same indices (clamp to handle edge cases)
                valid_rand_idx = rand_idx.clamp(max=idx_bank.shape[0] - 1)
                idx_sampled = idx_bank[valid_rand_idx].to(device=device)
                dataset_indices[idx_mask] = idx_sampled
    
        return samples, dataset_indices