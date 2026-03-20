## environment 
### conda <br>
```
conda create -n wire python=3.9.21
conda activate wire
```
### pytorch 2.0.1 <br>
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```
---

### pytorch3d-v0.7.4 installation <br>
Refer to [pytorch3d v0.7.4 github-install](https://github.com/facebookresearch/pytorch3d/blob/v0.7.4/INSTALL.md)

```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```
[anaconda-pytorch3d-v0.7.4 file](https://anaconda.org/pytorch3d/pytorch3d/files?page=1&version=0.7.4) if needed

---

### requirements:<br>
```
pip install -r requirements.txt
```
<br>
<br>

---
# Environment for wsl <br>
Refer to https://blog.csdn.net/m0_46349114/article/details/137602382
```
sudo apt update && sudo apt upgrade
sudo apt install build-essential
```
### download cuda toolkit 11.8 <br>
Link: [CUDA Toolkit 11.8 Downloads](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=runfile_local)
```
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```
### Environment setting
```
vim ~/.bashrc
export PATH=/usr/local/cuda/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
:wq
source ~/.bashrc
```
```
nvcc -V
```
### miniconda installation <br> 
refer to [CUDA Toolkit 11.8 Downloads](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=runfile_local)

### pytorch 2.0.1
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### pytorch3d-v0.7.7 installation <br>
Refer to [pytorch3d v0.7.7 github-install](https://github.com/facebookresearch/pytorch3d/blob/v0.7.7/INSTALL.md)

```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```
[anaconda-pytorch3d-v0.7.7 file](https://anaconda.org/pytorch3d/pytorch3d/files?page=1&version=0.7.7) if needed

### requirements:<br>
```
pip install -r requirements.txt
```