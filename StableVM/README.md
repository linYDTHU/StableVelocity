<h1 align="center">Stable Velocity Matching & <br> Variance-aware REPresentation Alignment</h1>

## 1. Environment Setup

```bash
conda create -n stablevm python=3.12 -y
conda activate stablevm
pip install -r requirements.txt
```

## 2. Dataset


We currently support experiments on [ImageNet](https://www.image-net.org/). You may place the dataset in any location and preprocess it using the tools in the `preprocessing` directory. Please refer to our [preprocessing guide](preprocessing/) for detailed instructions.

## 3. Training

```bash
accelerate launch train.py \
--report-to=wandb \
--allow-tf32 \
--mixed-precision=fp16 \
--seed=0 \
--path-type=linear \
--prediction=v \
--weighting=uniform \
--model=SiT-XL/2 \
--output-dir=exps \
--exp-name=linear-stablevm-bank=256-repa-sigmoid-t=0.7-k=20 \
--data-dir=Your_Data_Path \
--loss-type=stablevm \
--bank-capacity-per-class=256 \
--prefill-bank-fully \
--max-train-steps=2000000 \
--use-proj-loss \
--proj-weight-schedule=sigmoid \
--proj-k=20.0 \
--proj-tau=0.7 
```

The script will automatically create a folder in `exps` to save logs and checkpoints. The following options can be adjusted:

- `--loss-type`: Choose between CFM (`"si"`) and StableVM (`"stablevm"`).
- `--bank-capacity-per-class`: Memory bank capacity per class for StableVM when `--loss-type=="stablevm"` (default: 256).
- `--prefill-bank-fully`: If enabled, prefills the memory bank to full capacity per class before training; otherwise ensures at least one entry per class.
- `--use-proj-loss`: Enable representation alignment loss.
- `--proj-coeff`: Coefficient for projection loss (any value > 0).
- `--proj-weight-schedule`: Weighting schedule for projection loss. Options: `["hard", "hard_high", "sigmoid", "cosine", "snr"]`.
- `--proj-tau`: Split point (τ) for projection loss weighting (default: 0.7).
- `--proj-k`: Temperature/sharpness parameter for sigmoid schedule (default: 20).
- `--use-irepa`: Enable iREPA improvements: Conv projector and spatial normalization on encoder features.
- `--irepa-gamma`: Gamma coefficient for iREPA spatial normalization (default: 0.6).

## 4. Evaluation

Generate images using the following script. The output `.npz` file can be used with the [ADM evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations):

```bash
torchrun --nnodes=1 --nproc_per_node=4 generate.py \
  --model SiT-XL/2 \
  --num-fid-samples 50000 \
  --ckpt Your_Checkpoint_Path \
  --path-type=linear \
  --encoder-depth=8 \
  --projector-embed-dims=768 \
  --per-proc-batch-size=64 \
  --mode=sde \
  --num-steps=250 \
  --cfg-scale=1.8 \
  --guidance-high=0.7 \
  --use-projector
```

The following options can be adjusted:

- `--use-projector`: Include a projector in the SiT model.
- `--use-irepa`: Use the CONV projection layer from [iREPA](https://github.com/End2End-Diffusion/iREPA).
- `--label-sampling`: Label sampling strategy. `'equal'` ensures each class appears equally; `'random'` samples uniformly.

## 5. Toy Experiments (GMM)

We provide toy experiments on 2D Gaussian Mixture Models (GMM) to demonstrate the variance perspective of flow matching. The `gmm_exp.py` script compares CFM, STF, and StableVM methods using the same model architecture under linear interpolation.

### Basic Usage

```bash
python gmm_exp.py \
  --exp stable_vm \
  --vector_field velocity
```

### Options

- `--exp` / `-e`: Experiment type. Choose from `["cfm", "stf", "stable_vm", "stable_vm_efficient"]`:
  - `cfm`: Conditional Flow Matching (baseline)
  - `stf`: Stable Target Field (using Gaussian perturbation)
  - `stable_vm`: Stable Velocity Matching (using GMM perturbation)
  - `stable_vm_efficient`: Memory-efficient version of StableVM
- `--vector_field`: Vector field type. Choose from `["velocity", "score"]`.
- `--use_wandb`: Enable Weights & Biases logging.
- `--no_plot_variance`: Disable variance curve plotting.
- `--only_plot`: Plot variance curves without training.
- `--enable_visualization`: Enable vector field visualization (generates GIFs).
- `--imagenet-data-path`: Path to ImageNet dataset for variance curve evaluation (optional).

### Examples

**Plot variance curves only (no training):**
```bash
python gmm_exp.py \
  --vector_field velocity \
  --only_plot
```

**Compare CFM vs StableVM:**
```bash
# Run CFM
python gmm_exp.py --exp cfm --vector_field velocity

# Run StableVM
python gmm_exp.py --exp stable_vm --vector_field velocity
```

The script performs the following:
- Generates variance curves comparing CFM and StableVM targets at different timesteps
- Trains a neural network to learn the vector field
- Evaluates Fisher divergence between predicted and true fields
- Optionally visualizes vector fields and saves comparison GIFs
- Saves evaluation results in `plots/data/` for further analysis

## Note

While we have made every effort to ensure code quality, there may be discrepancies between the code and the results reported in the paper due to potential errors during code preparation and release. If you encounter difficulties reproducing our findings, please do not hesitate to contact us. We plan to conduct additional sanity-check experiments in the near future.

## Acknowledgement

This codebase is primarily built upon [REPA](https://github.com/sihyun-yu/REPA) and [iREPA](https://github.com/End2End-Diffusion/iREPA). We also follow the implementation of the balanced sampling strategy from [REPA-E](https://github.com/End2End-Diffusion/REPA-E).
