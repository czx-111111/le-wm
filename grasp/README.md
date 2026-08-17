# GRASP: Parallel stochastic planning for world models

This repository adds the **GRASP** planner – a gradient-based planner with virtual latent states and stochastic state updates – for learned world models. GRASP optimizes both actions and intermediate “virtual” states in a lifted trajectory space, using dynamics consistency penalties to stay faithful to the learned dynamics while enabling efficient, parallelized planning.

Concretely, GRASP optimizes a lifted objective over virtual states $\mathbf{s}$ and actions $\mathbf{a}$ of the form

```math
\mathcal{L}(\mathbf{s}, \mathbf{a}) =
\sum_{t=0}^{T-1} \big\| F_\theta(\bar{s}_t, a_t) - s_{t+1} \big\|_2^2
\;+\;
\gamma \sum_{t=0}^{T-1} \big\| F_\theta(\bar{s}_t, a_t) - g \big\|_2^2 ,
```

where $F_\theta$ is the learned world model, $\bar{s}_t$ is a stop-gradient copy of $s_t$ (no gradients through the state input), and $g$ is the encoded goal state. The first term enforces pairwise dynamics consistency between virtual states, while the second provides dense goal shaping to counteract the stopgrad's initial state biased reshaping.

An example planning command with GRASP is:

```bash
python plan.py model_name=pusht n_evals=5 planner=grasp goal_H=8
```

## Minimal pseudocode: `grasp_pseudocode.py`

A self-contained, minimal implementation of the core GRASP algorithm lives in **`grasp_pseudocode.py`** at the repository root, for ease of re-implementation. This file strips away the full planner’s infrastructure (logging, CEM, Lagrangian terms, mixed precision, etc.) and exposes only the essential method.

**What it provides**

- **World model interface**: A clear specification of what a world model must implement—a single `rollout(s_0, a)` function that outputs the state sequence from initial state s_0 over actions a.
- **Core algorithm**: Virtual state initialization (linear interpolation), the stop-gradient dynamics + goal-shaping loss, Langevin-style state noise, and periodic full-rollout sync.
- **Runnable example**: A simple integrator world model (`s_next = s + dt * a`) and a `__main__` block that runs a barebones version of GRASP.


## Built on DINO-WM

This codebase is a derivative of the **DINO-WM** project, reusing its world-model training, datasets, and environment wrappers while adding GRASP-focused planning algorithms and analysis.

- **Paper**: [[DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning]](https://arxiv.org/abs/2411.04983)  
- **Data**: [OSF dataset bucket](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28)  
- **Project website**: [https://dino-wm.github.io/](https://dino-wm.github.io/)

If you use this repository, please also cite the original DINO-WM paper.


## More Environments and World Models

GRASP can be used with other world models beyond DINO-WM. We provide **adapter repos** that integrate GRASP with additional world model codebases (and will PR once stable):

| Adapter | World Model | Environments |
|---------|-------------|-------------|
| [le-wm-grasp](https://github.com/michael-psenka/le-wm-grasp) | LeWorldModel (LeWM) | PushT, TwoRoom, Reacher, Cube |
| [jepa-wms-grasp](https://github.com/michael-psenka/jepa-wms-grasp) | JEPA-WMs | Metaworld, DROID, RoboCasa |

To set up an adapter, use the `setup_adapter.sh` script:

```bash
# Clone a specific adapter into adapters/<name>
bash setup_adapter.sh le-wm-grasp

# Clone to a custom directory
bash setup_adapter.sh jepa-wms-grasp /path/to/my/adapter

# Clone all available adapters
bash setup_adapter.sh all
```

Each adapter repo contains its own installation instructions, pretrained model links, and GRASP configuration. The `adapters/` directory is gitignored so the base repo stays clean.

## How GRASP works (high level)

At a high level, GRASP is a **parallel stochastic gradient-based planner for world models** with four key ideas:

- **Lifted virtual states**: Instead of optimizing only actions through a long serial rollout, GRASP introduces intermediate **virtual states** $s_1,\dots,s_{T-1}$ and penalizes pairwise dynamics violations $\|F_\theta(s_t,a_t) - s_{t+1}\|^2$. This makes all world-model evaluations across time **parallelizable**.
- **State-space exploration via noise**: During optimization, GRASP updates states with a descent-like step **plus Gaussian noise**. This encourages exploration in the lifted state space and helps escape bad local minima while keeping actions guided by gradients.
- **Cut brittle state-input gradients, keep action gradients**: Gradients through the state inputs of high-dimensional vision models can be adversarial. GRASP uses **stop-grad copies** of states $\bar{s}_t$ inside the world model so that gradients flow cleanly to actions but not through fragile $\nabla_s F_\theta$ directions.
- **Goal shaping and periodic rollout sync**: A dense **goal-shaping term** encourages each predicted next state to move toward the goal, and every few iterations GRASP runs a short **serial rollout sync** step that takes small gradient steps on the original terminal loss, refining actions without losing the benefits of the smoother lifted landscape.

Together, these ingredients yield a planner that is (1) parallel over time during exploration, (2) robust to brittle state gradients, and (3) periodically grounded by true rollout-based gradients.

## Getting started

1. [Installation](#installation)
2. [Datasets](#datasets)
3. [Train a world model](#train-a-world-model)
4. [Plan with GRASP](#plan-with-grasp)
5. [Analyze results](#analyze-results)

## Installation

The easiest way to set everything up is via the `install.sh` script in the repository root. By default it:

- creates/updates the `grasp` conda environment from `environment.yaml`,
- can optionally install Mujoco,
- configures `DATASET_DIR` and can download datasets from OSF,
- and can extract PushT frames.

The following will set everything up from scratch:

```bash
git clone https://github.com/michael-psenka/dino-wm-admm-planning.git
cd dino-wm-admm-planning

# Full setup: env + Mujoco + dataset dir + download datasets + extract PushT frames
./install.sh --all --dataset-dir /path/to/base
```

This will also download the **default pre-trained world models** (for `point_maze`, `wall_single`, and `pusht`) into `<base>/outputs`, which matches where `plan.py` loads from when `ckpt_base_path` points to your base directory. If you prefer to **train your own world model**, you can either:

- run `./install.sh --conda` to just set up the environment, then use `train.py` manually, or  
- combine `--dataset-dir` / `--download-datasets` with your own training commands (see [Train a world model](#train-a-world-model)).

If you want to set things up manually instead of using `install.sh`, you can still clone the repo and create the environment directly:

```bash
git clone https://github.com/michael-psenka/dino-wm-admm-planning.git
cd dino-wm-admm-planning
conda env create -f environment.yaml
conda activate grasp
```

### Mujoco

If you pass `--mujoco` (or `--all`) to `install.sh`, it will install Mujoco 2.1.0 under `~/.mujoco` and add the appropriate entries to `LD_LIBRARY_PATH`. You can also install Mujoco manually with:

```bash
mkdir -p ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P ~/.mujoco/
cd ~/.mujoco
tar -xzvf mujoco210-linux-x86_64.tar.gz
```

Append the following lines to your `~/.bashrc` (adjust the username and NVIDIA path as needed):

```bash
# Mujoco path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<username>/.mujoco/mujoco210/bin

# NVIDIA library path (if using NVIDIA GPUs)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

Reload your shell configuration:

```bash
source ~/.bashrc
```

#### Notes

- **GPU drivers**: ensure NVIDIA drivers are correctly installed for GPU-accelerated simulations.
- **Debugging paths**: if Mujoco fails to load, double-check `LD_LIBRARY_PATH`.

## Datasets

If you pass `--dataset-dir /path/to/base` and `--download-datasets` (or `--all`) to `install.sh`, it will:

- set `DATASET_DIR=<base>/datasets` (persisted in your shell config),
- download missing datasets from OSF,
- and place pre-trained model directories under `<base>/outputs` (where `plan.py` loads from via `ckpt_base_path/outputs`).

Datasets for the supported tasks can also be downloaded directly from a public OSF bucket:  
`https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28`

If you prefer, you can use `install.sh` just to set up `DATASET_DIR` and optionally download datasets:

```bash
# From the repository root
./install.sh --dataset-dir /path/to/base

# Or to download and extract datasets automatically:
./install.sh --dataset-dir /path/to/base --download-datasets
```

After setup, `DATASET_DIR` points to `<base>/datasets`. Inside that folder you should see:

```text
<base>/datasets/
├── point_maze
├── pusht          # raw videos in pusht/train/obses, pusht/val/obses; run --extract-pusht-frames to extract frames
└── wall_single
```

Pre-trained model directories (if downloaded) are stored under `<base>/outputs/`. The install script sets `CKPT_BASE_PATH=<base>` in your shell config, which `plan.py` uses by default; models load from `<base>/outputs/<model_name>/`. If you set up manually, export `CKPT_BASE_PATH` or override `ckpt_base_path` in `conf/plan_grasp.yaml`.

## Train a world model

Training follows the existing configuration structure. From the repo root:

```bash
conda activate grasp
python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3
```

You can control where outputs are written via `ckpt_base_path` in `conf/train.yaml`. Checkpoints will be saved under `${ckpt_base_path}/outputs`.

## Plan with GRASP

Once you have a trained world model, you can run GRASP-based planning with:

```bash
conda activate grasp
python plan.py model_name=<model_name> n_evals=5 planner=grasp goal_H=5 goal_source='random_state' planner.opt_steps=30
```

- `model_name` should match the checkpoint directory name under `<ckpt_base_path>/outputs`.
- `goal_H` controls how far away the sampled goal is (in steps).
- Other planners (CEM, gradient descent variants, LATCO, etc.) are available as baselines via `planner=<name>`.

For more advanced configurations (e.g., custom output directories, SLURM runs, or sweeps), see:

- `conf/plan_grasp.yaml` (default GRASP planning config used by `plan.py`)
- `conf/plan.yaml` (generic planning config)
- `conf/planner/grasp.yaml` (GRASP hyperparameters)

## Analyze results

Analysis utilities live under `analysis/`:

- `analysis/experiment_results.py`: loading and aggregating `logs.json` outputs from `plan.py`.
- `analysis/visualize_results.py`: scripts for plotting GRASP vs baseline planners, sweeping over goal horizon, or analyzing GRASP hyperparameter sweeps (stored under `grasp` directory names in the baselines layout).

The tests in `tests/test_experiment_results.py` exercise these utilities against fixtures in `tests/fixtures/` and baselines stored under `baselines/`.

## Repository overview

This repository is organized around three main components:

- **World model training**: `train.py` with configs under `conf/train*.yaml` and model/env configs in `conf/encoder`, `conf/decoder`, etc.
- **Planning**: `plan.py` together with planner configs in `conf/planner` (notably `conf/planner/grasp.yaml`) and environment configs in `conf/env`.
- **Analysis and baselines**: utilities under `analysis/` and pre-computed baselines under `baselines/`.

Key top-level directories:

- `planning/`: planning algorithms and evaluators (including the GRASP planner in `planning/grasp.py`).
- `env/`: environment registrations and wrappers for tasks such as point_maze, wall, pusht, ballnav, and random_mlp.
- `models/`: visual world models and component encoders/decoders built on top of pre-trained vision backbones.
- `metrics/`: evaluation metrics (e.g., LPIPS in `metrics/lpipsPyTorch`) and image-based metrics.
- `analysis/`: helpers for loading and aggregating experiment results.
- `conf/`: Hydra configuration tree for training, planning, environments, and model components.
- `tests/`: pytest-based tests for planners and experiment result loading.
- `baselines/`: stored `logs.json` files for baseline and ablation experiments.
- `distributed_fn/`: utilities for launching distributed training jobs.

## Typical workflows

- **Use a pre-trained world model (no retraining)**
  - Run `install.sh` with `--dataset-dir` and `--download-datasets` (or `--all`) to download datasets and pre-trained models for `point_maze`, `wall_single`, and `pusht`.
  - Then go directly to planning:
    ```bash
    conda activate grasp
    python plan.py model_name=pusht n_evals=5 planner=grasp goal_H=5 goal_source='random_state' planner.opt_steps=30
    ```

- **OR: Train a world model**
  - If you want to train from scratch or fine-tune:
    ```bash
    conda activate grasp
    python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3
    ```
  - You can then pass your new `model_name` to `plan.py` as above.

- **Plan with a trained world model using GRASP**
  - After training or choosing a pre-trained checkpoint, run:
    ```bash
    conda activate grasp
    python plan.py model_name=<model_name> n_evals=5 planner=grasp goal_H=5 goal_source='random_state' planner.opt_steps=30
    ```

- **Analyze experiment results**
  - Use `analysis/experiment_results.py` and `analysis/visualize_results.py` to aggregate and plot GRASP vs baseline performance.



## Citation

```text
@article{psenka2026grasp,
      title={Parallel Stochastic Gradient-Based Planning for World Models},
      author={Michael Psenka and Michael Rabbat and Aditi Krishnapriyan and Yann LeCun and Amir Bar},
      year={2026},
      eprint={2602.00475},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.00475},
}
```
