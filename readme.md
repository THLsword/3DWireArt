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

The `data_util/` folder contains a few optional preprocessing scripts:

- `data_util/3Dmodel_preprocessing/obj_preprocess.py`: if your input is an `.obj` mesh, you can use this script to sample points from the mesh surface, normalize them, and generate `model_normalized_4096.npz` under `data/models/<model_name>/`.
- `data_util/template_preprocessing/preprocess_obj.py`: if you are preparing a new template mesh, you can use this script to normalize the template and generate `vertices.txt`, `topology.txt`, and `adjacencies.txt`.
- `data_util/template_preprocessing/preprocess_symmetry.py`: if your template is symmetric, you can use this script to generate `symmetries.txt` from the processed template vertices.

## How To Use

Run the scripts in this order:

```bash
bash scripts/preprocess.sh
bash scripts/train.sh
bash scripts/postprocess.sh
```
