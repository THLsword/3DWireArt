# WireArtFitting

## How To Start

### 1. Create the environment

```bash
conda create -n wire python=3.9.21
conda activate wire
```

### 2. Install PyTorch

```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

### 3. Install PyTorch3D

Follow the official installation guide for PyTorch3D v0.7.4:

- https://github.com/facebookresearch/pytorch3d/blob/v0.7.4/INSTALL.md

Conda installation:

```bash
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```

If you need prebuilt package information:

- https://anaconda.org/pytorch3d/pytorch3d/files?page=1&version=0.7.4

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

## Data Preparation

Put each input model under `data/models/<model_name>/`.

The current scripts assume the point cloud file is:

```text
data/models/<model_name>/model_normalized_4096.npz
```

## How To Use

The runnable shell scripts are stored in `scripts/`:

- `scripts/preprocess.sh`
- `scripts/train.sh`
- `scripts/post_perceptual.sh`

Before running them, edit the variables at the top of each script, especially:

- `model_name`
- `epoch`
- `learning_rate`
- `template`

Run all commands from the repository root.

### 1. Preprocessing

Command:

```bash
bash scripts/preprocess.sh
```

What it does:

- Loads the input point cloud from `data/models/<model_name>/model_normalized_4096.npz`.
- Renders multi-view images of the point cloud.
- Computes alpha-shape contour images.
- Trains view-dependent weights for the point cloud.
- Saves preprocessing outputs to `outputs/<model_name>/prep_outputs/`.

Main outputs:

- `outputs/<model_name>/prep_outputs/render_outputs/`
- `outputs/<model_name>/prep_outputs/alpha_outputs/`
- `outputs/<model_name>/prep_outputs/train_outputs/weights.pt`
- `outputs/<model_name>/prep_outputs/train_outputs/multi_view.obj`

### 2. Training

Command:

```bash
bash scripts/train.sh
```

What it does:

- Loads the preprocessing result `weights.pt`.
- Loads the template and target point cloud.
- Optimizes the template control points to fit the target shape.
- Writes intermediate fitting results and the final control points.

Main outputs:

- `outputs/<model_name>/control_points.obj`
- `outputs/<model_name>/model_weights.pth`
- `outputs/<model_name>/save_opt/`
- `outputs/<model_name>/logs/`

### 3. Post-Perceptual Pruning

Command:

```bash
bash scripts/post_perceptual.sh
```

What it does:

- Loads the fitted control points from training.
- Evaluates curves with a multi-view perceptual criterion.
- Removes redundant curves while preserving connectivity.
- Exports the pruned wire structure for downstream use.

Main outputs:

- `outputs/<model_name>/post_outputs/`

## Recommended Pipeline

Run the scripts in this order:

```bash
bash scripts/preprocess.sh
bash scripts/train.sh
bash scripts/post_perceptual.sh
```

