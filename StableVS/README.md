# StableVS: Stable Velocity Sampling Extensions

**StableVS** provides a set of custom schedulers for the [diffusers](https://github.com/huggingface/diffusers) library, enabling **faster diffusion sampling with preserved generation quality** via *Stable Velocity Sampling* strategies.

The implementation targets both text‑to‑image (T2I) and text‑to‑video (T2V) pipelines and integrates seamlessly with existing Diffusers workflows.

## Overview

StableVS provides three custom schedulers:

1. **StableVSFlowMatchScheduler** - A modified Flow Match Euler Discrete Scheduler combined with StableVS
2. **StableVSDPMSolverMultistepScheduler** - A modified DPM-Solver++ scheduler combined with StableVS
3. **StableVSUniPCMultistepScheduler** - A modified UniPC multistep scheduler combined with StableVS

## Installation

### Prerequisites

```bash
# Create conda environment
conda create -n stablevs python=3.10 -y
conda activate stablevs

# Install PyTorch with CUDA support first (adjust cu121 to match your CUDA version)
# See https://pytorch.org/get-started/locally/ for other configurations
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Install StableVS

```bash
# Clone the repositories
git clone https://github.com/linYDTHU/StableVelocity.git

# 1. Install diffusers with test dependencies (includes transformers, sentencepiece, etc.)
pip install "diffusers[test]"

# 2. Install StableVS
cd StableVelocity/StableVS
pip install -e .
```

## Usage

### Basic Usage

```python
from stablevs import (
    StableVSFlowMatchScheduler,
    StableVSDPMSolverMultistepScheduler,
    StableVSUniPCMultistepScheduler
)
from diffusers import FluxPipeline

# Load your model
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)

# Use custom scheduler
pipe.scheduler = StableVSFlowMatchScheduler(
    num_train_timesteps=1000,
    use_fast_low_schedule=True,
    fast_low_split_point=0.85,
    fast_low_low_substeps=9,
    low_region_noise_factor=0.0,
)

# Generate image
image = pipe(
    prompt="a beautiful landscape",
    num_inference_steps=30,
    guidance_scale=3.5,
).images[0]
```

### Quick Demo

We provide simple demo scripts to compare baseline vs StableVS schedulers:

```bash
# Text-to-Image: compare Euler vs StableVS on SD3.5, Flux, Qwen-Image
python examples/t2i_demo.py --models sd35,flux,qwen --output-dir ./figures

# Text-to-Video: compare UniPC vs StableVS-UniPC on Wan2.2
python examples/t2v_demo.py --output-dir ./videos

# Print sigma schedules only (no GPU needed)
python examples/t2i_demo.py --print-sigmas-only
python examples/t2v_demo.py --print-sigmas-only
```

### Geneval benchmark (T2I)

We provide a generation script for [*Geneval*](https://github.com/djghosh13/geneval) benchmark on multiple T2I models:

```bash
python examples/geneval_generate.py \
    --prompts generation_prompts.txt \
    --output-dir ./samples \
    --models flux,sd35,qwen \
    --num-steps 30 \
    --sampler custom_euler \
    --fast-low-split-point 0.85 \
    --fast-low-low-substeps 9 \
    --low-region-noise-factor 0.0
```

Supported models:
- `flux` - FLUX.1-dev
- `sd3` - Stable Diffusion 3 Medium
- `sd35` - Stable Diffusion 3.5 Large
- `qwen` - Qwen Image 2512

Supported samplers:
- `euler` - Default scheduler
- `custom_euler` - StableVSFlowMatchScheduler
- `dpm` - DPMSolverMultistepScheduler
- `custom_dpm` - StableVSDPMSolverMultistepScheduler

The script automatically detects available GPUs and performs parallel sampling.

### T2V-CompBench

For [*T2V-CompBench*](https://github.com/KaiyueSun98/T2V-CompBench) evaluation:

```bash
python examples/t2v_compbench_generate.py \
    --save_dir ./video/custom_30steps \
    --prompts_dir [Path/T2V-CompBench/prompts]\
    --sampler custom \
    --num_steps 30 \
    --multi_gpu
```

Supported samplers:
- `unipc` - Default scheduler
- `custom` - StableVSUniPCMultistepScheduler

## Scheduler Parameters

### Fast-Low Schedule Parameters

All custom schedulers support the following fast-low schedule parameters:

- `use_fast_low_schedule` (bool): Enable fast-low schedule
- `fast_low_split_point` (float): Split point between high and low regions (default: 0.85)
- `fast_low_low_substeps` (int): Number of substeps in low region (default: 9)
- `low_region_noise_factor` (float): Noise factor for low region (default: 0.0)

## Project Structure

```
StableVS/
├── stablevs/
│   ├── __init__.py
│   └── schedulers/
│       ├── __init__.py
│       ├── scheduling_stablevs_dpmsolver_multistep.py
│       ├── scheduling_stablevs_flow_match.py
│       └── scheduling_stablevs_unipc_multistep.py
├── examples/
│   ├── t2i_demo.py                # T2I quick demo (SD3.5, Flux, Qwen)
│   ├── t2v_demo.py                # T2V quick demo (Wan2.2)
│   ├── geneval_generate.py        # Batch T2I generation script
│   └── t2v_compbench_generate.py  # T2V-CompBench generation script
├── setup.py
├── generation_prompts.txt         # prompts for Geneval
└── README.md
```

## Acknowledgments

This project is built on top of the [diffusers](https://github.com/huggingface/diffusers) library by Hugging Face.
