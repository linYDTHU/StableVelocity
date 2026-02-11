<h1 align="center">Stable Velocity:<br>A Variance Perspective on Flow Matching</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2602.05435">
    <img src="https://img.shields.io/badge/arXiv-2602.05435-b31b1b.svg" alt="arXiv">
  </a>
  &nbsp;
  <img src="https://visitor-badge.laobi.icu/badge?page_id=linydthu.StableVelocity" alt="visitors">
</p>

<p align="center">
  <a href="https://linydthu.github.io/" target="_blank">Donglin&nbsp;Yang</a><sup>1,3</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=dFsd0owAAAAJ&hl=en" target="_blank">Yongxing&nbsp;Zhang</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://xinyu-andy.github.io/" target="_blank">Xin&nbsp;Yu</a><sup>2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://liang-hou.github.io/" target="_blank">Liang&nbsp;Hou</a><sup>3</sup>
  <br>
  <a href="https://www.xtao.website/" target="_blank">Xin&nbsp;Tao</a><sup>3</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://magicwpf.github.io/" target="_blank">Pengfei&nbsp;Wan</a><sup>3</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://xjqi.github.io/" target="_blank">Xiaojuan&nbsp;Qi</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://lrjconan.github.io/" target="_blank">Renjie&nbsp;Liao</a><sup>2</sup>
  <br><br>
  <sup>1</sup> HKU &emsp; <sup>2</sup> UBC &emsp; <sup>3</sup> Kling Team, Kuaishou Technology
</p>

<h3 align="center">[<a href="https://linydthu.github.io/StableVelocity/">project page</a>]&emsp;[<a href="https://arxiv.org/abs/2602.05435">arXiv</a>]</h3>

---

> **Summary** &mdash; By explicitly characterizing the variance of flow matching, we identify 1) a *high-variance regime* near the prior, where optimization is challenging, and 2) a *low-variance regime* near the data distribution, where conditional and marginal velocities nearly coincide. Leveraging this insight, we propose **Stable Velocity**, a unified framework that improves both training and sampling. For training, we introduce Stable Velocity Matching (StableVM), an unbiased variance-reduction objective, along with Variance-Aware Representation Alignment (VA-REPA), which adaptively strengthen auxiliary supervision in the *low-variance regime*. For inference, we show that dynamics in the *low-variance regime* admit closed-form simplifications, enabling Stable Velocity Sampling (StableVS), a finetuning-free acceleration.

---

## 🔍 Motivation: Variance Regimes in Flow Matching

Conditional Flow Matching (CFM) trains neural velocity fields using single-sample conditional velocities. While unbiased, these targets can exhibit **high variance**, especially when the marginal distribution remains close to the prior. We empirically observe that this variance is **highly non-uniform over time**.

<p align="center">
  <img src="assets/images/variance_curves.png" width="85%">
</p>

Flow Matching naturally decomposes into two regimes:

- **Low-variance regime (near data)** — The posterior concentrates on a single reference sample, and conditional and marginal velocities nearly coincide.

- **High-variance regime (near prior)** — The posterior spreads over multiple samples, leading to large variance in conditional velocity targets.

This regime structure becomes more pronounced in high-dimensional data, such as ImageNet latents. Why does this happen?

<p align="center">
  <img src="assets/images/variance_demo.png" width="85%">
</p>

## 🚀 Stable Velocity: A Variance-Driven Framework

Stable Velocity is a unified framework that leverages this variance structure to improve both **training** and **sampling**.

### 1. Stable Velocity Matching (StableVM)

StableVM replaces the single-sample target with a **multi-sample, self-normalized estimator** under **multi-sample conditional path**.

- Unbiased estimator of the true marginal velocity
- Strictly lower variance than standard CFM
- Compatible with general stochastic interpolants

### 2. Variance-Aware Representation Alignment (VA-REPA)

Representation alignment methods (e.g., REPA) are effective only when the noisy input retains semantic information. From a variance perspective, this occurs **exclusively in the low-variance regime**.

<p align="center">
  <img src="assets/images/regime_aware_repa_motivation.png" width="85%">
</p>

Applying representation alignment uniformly along the diffusion trajectory introduces noisy supervision. VA-REPA activates alignment **only in the low-variance regime**, leading to consistent improvements over REPA as well as its variants.

### 3. Stable Velocity Sampling (StableVS)

In the low-variance regime, the probability flow dynamics admit **closed-form simplifications**. StableVS exploits this structure to enable **finetuning-free acceleration** of pretrained models, achieving more than **2× faster sampling** in the low-variance regime without perceptible degradation in sample quality.

#### StableVS on SD3.5 (Text-to-Image)

**Prompt:** *"A turquoise river winds through a lush canyon. Thick moss and dense ferns blanket the rocky walls; multiple waterfalls cascade from above, enveloped in mist. At noon, sunlight filters through the dense canopy, dappling the river surface with shimmering light. The atmosphere is humid and fresh, pulsing with primal jungle vitality. No humans, text, or artificial traces present."*

<table align="center">
  <tr>
    <th>Euler (30 steps)</th>
    <th>Euler (20 steps)</th>
    <th>Euler (11) + StableVS (9)</th>
  </tr>
  <tr>
    <td><img src="assets/images/sd35_euler_30steps.jpg" width="240"></td>
    <td><img src="assets/images/sd35_euler_20steps.jpg" width="240"></td>
    <td><img src="assets/images/sd35_stablevs_20steps.jpg" width="240"></td>
  </tr>
</table>

#### StableVS on Flux (Text-to-Image)

**Prompt:** *"A cat holding a sign that says Stable Velocity"*

<table align="center">
  <tr>
    <th>Euler (30 steps)</th>
    <th>Euler (20 steps)</th>
    <th>Euler (11) + StableVS (9)</th>
  </tr>
  <tr>
    <td><img src="assets/images/flux_euler_30steps.jpg" width="240"></td>
    <td><img src="assets/images/flux_euler_20steps.jpg" width="240"></td>
    <td><img src="assets/images/flux_stablevs_20steps.jpg" width="240"></td>
  </tr>
</table>

#### StableVS on Qwen-Image (Text-to-Image)

**Prompt:** *"A 20-year-old East Asian girl with delicate, charming features and large, bright brown eyes—expressive and lively, with a cheerful or subtly smiling expression. Her naturally wavy long hair is either loose or tied in twin ponytails. She has fair skin and light makeup accentuating her youthful freshness. She wears a modern, cute dress or relaxed outfit in bright, soft colors—lightweight fabric, minimalist cut. She stands indoors at an anime convention, surrounded by banners, posters, or stalls. Lighting is typical indoor illumination—no staged lighting—and the image resembles a casual iPhone snapshot: unpretentious composition, yet brimming with vivid, fresh, youthful charm."*

<table align="center">
  <tr>
    <th>Euler (30 steps)</th>
    <th>Euler (17 steps)</th>
    <th>Euler (8) + StableVS (9)</th>
  </tr>
  <tr>
    <td><img src="assets/images/qwen_euler_30steps.jpg" width="240"></td>
    <td><img src="assets/images/qwen_euler_17steps.jpg" width="240"></td>
    <td><img src="assets/images/qwen_stablevs_17steps.jpg" width="240"></td>
  </tr>
</table>

#### StableVS on Wan2.2 (Text-to-Video)

**Prompt:** *"Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."*

<table align="center">
  <tr>
    <th>UniPC (30 steps)</th>
    <th>UniPC (20 steps)</th>
    <th>UniPC (11) + StableVS (9)</th>
  </tr>
  <tr>
    <td><img src="assets/gifs/boxing_unipc_30steps.gif" width="240"></td>
    <td><img src="assets/gifs/boxing_unipc_20steps.gif" width="240"></td>
    <td><img src="assets/gifs/boxing_stablevs_20steps.gif" width="240"></td>
  </tr>
</table>

**Prompt:** *"A horse jumps over a fence."*

<table align="center">
  <tr>
    <th>UniPC (30 steps)</th>
    <th>UniPC (20 steps)</th>
    <th>UniPC (11) + StableVS (9)</th>
  </tr>
  <tr>
    <td><img src="assets/gifs/horse_jump_unipc_30steps.gif" width="240"></td>
    <td><img src="assets/gifs/horse_jump_unipc_20steps.gif" width="240"></td>
    <td><img src="assets/gifs/horse_jump_stablevs_20steps.gif" width="240"></td>
  </tr>
</table>


## ⚡ Quick Start (StableVM & VA-REPA)

```bash
cd StableVM
conda create -n stablevm python=3.12 -y
conda activate stablevm
pip install -r requirements.txt
```

**Training** — Launch StableVM + VA-REPA:

```bash
accelerate launch train.py --model=SiT-XL/2 --data-dir=/path/to/data \
    --loss-type=stablevm --bank-capacity-per-class=256 --prefill-bank-fully \
    --use-proj-loss --proj-weight-schedule=sigmoid --proj-tau=0.7 \
    --allow-tf32 --mixed-precision=fp16
```

**Evaluation** — Generate 50K samples for FID:

```bash
torchrun --nnodes=1 --nproc_per_node=4 generate.py --model SiT-XL/2 \
    --ckpt /path/to/checkpoint.pt --num-fid-samples 50000 --use-projector \
    --mode=sde --num-steps=250 --cfg-scale=1.8 --guidance-high=0.7
```

📖 For the full list of options, toy GMM experiments, and more details, see the [StableVM README](StableVM/).

## ⚡ Quick Start (StableVS)

StableVS provides drop-in custom schedulers for the [diffusers](https://github.com/huggingface/diffusers) library, enabling **finetuning-free** accelerated sampling on pretrained models.

```bash
cd StableVS
conda create -n stablevs python=3.10 -y
conda activate stablevs

# Install PyTorch (adjust cu121 to match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install diffusers and StableVS
pip install "diffusers[test]"
pip install -e .
```

**Basic Usage** — Replace the scheduler in any diffusers pipeline:

```python
import torch
from stablevs import StableVSFlowMatchScheduler
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.scheduler = StableVSFlowMatchScheduler(
    num_train_timesteps=1000,
    use_fast_low_schedule=True,
    fast_low_split_point=0.85,
    fast_low_low_substeps=9,
    low_region_noise_factor=0.0,
)
image = pipe("a beautiful landscape", num_inference_steps=30, guidance_scale=3.5).images[0]
```

**Quick Demos** — Compare baseline vs StableVS schedulers:

```bash
# Text-to-Image (SD3.5, Flux, Qwen-Image)
python examples/t2i_demo.py --models sd35,flux,qwen --output-dir ./figures

# Text-to-Video (Wan2.2)
python examples/t2v_demo.py --output-dir ./videos

# Print sigma schedules only (no GPU needed)
python examples/t2i_demo.py --print-sigmas-only
python examples/t2v_demo.py --print-sigmas-only
```

📖 For the full list of supported schedulers, benchmark scripts, and parameter details, see the [StableVS README](StableVS/).

## 📚 Citation

If you find our paper or code useful, please consider citing our paper:

```bibtex
@misc{yang2026stablevelocityvarianceperspective,
      title={Stable Velocity: A Variance Perspective on Flow Matching}, 
      author={Donglin Yang and Yongxing Zhang and Xin Yu and Liang Hou and Xin Tao and Pengfei Wan and Xiaojuan Qi and Renjie Liao},
      year={2026},
      eprint={2602.05435},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.05435}, 
}
```

## ✅ Checklist

- [x] ~~StableVM and VA-REPA code release~~
- [ ] Model checkpoints for StableVM and VA-REPA
- [x] ~~StableVS code release~~
- [x] ~~Project page release~~

## 👨🏻‍💻 Contact

Feel free to contact [Donglin Yang](mailto:ydlin718@gmail.com) or submit a GitHub issue if you have identified any bugs.
