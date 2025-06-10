# Processing Environment Setup

This guide sets up a virtual environment on HPC to perform batch inference on binary image data. 


# Create Connection to HPC Cluster
TreeOfLife is processed on Ohio Supercomputer Center (OSC). We'll use OSC's [Ascend cluster](https://www.osc.edu/resources/technical_support/supercomputers/ascend) as example.

``` bash
ssh ascend.osc.edu
```

# Load Essential Modules

In this step we're going to load specific version of Python & CUDA, which will dictate the PyTorch version. 

You might have different Python or CUDA version avaiable on your computing platform, please refer to [PyTorch website](https://pytorch.org/) to install the corresponding PyTorch version. Note that the latest PyTorch 2.7.0 requires Python 3.9 or later. 

``` bash
module load python/3.12
module load cuda/12.4.1
```

Print out version to verify
``` bash
python --version
```
```
Python 3.12.4
```
``` bash
nvcc --version
```
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0
```

# Create Virtual Environment with `uv`

Make sure `uv` is installed first:
``` bash
pip install uv
```


# CLIP

``` bash
uv venv batch_clip
source batch_clip/bin/activate  # Linux OS

# Verify python version
which python
```
If you have:
- Linux OS
- Python 3.12
- CUDA 12.4.1

Consider install dependencies from `requirements_batch_clip.txt` 
``` bash 
uv pip install requirements_batch_clip.txt
```

Otherwise, please follow the section below to re-create the virtual environment.

## Install `torch`
``` bash
uv pip install torch==2.7.0 torchvision torchaudio
```

## Install `clip`

Install [CLIP](https://github.com/openai/CLIP)

``` bash 
uv pip install ftfy regex tqdm
uv pip install git+https://github.com/openai/CLIP.git
```

## Install Utility Packages

``` bash
uv pip install pandas pyarrow pyspark Pillow 
```
Install other utility packages as needed.

## Confirm Installation

Test from command line:
``` bash
python -c "import torch; import clip; print('torch:', torch.__version__); print('clip loaded')"
```

Test from scripts
``` python
import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

print(f"Model loaded on device: {device}")
```

# Face Detection

``` bash
uv venv batch_face_detection
source batch_face_detection/bin/activate  # Linux OS

# Verify python version
which python
```
If you have:
- Linux OS
- Python 3.12
- CUDA 12.4.1

Consider install dependencies from `requirements_batch_face_detection.txt` 
``` bash 
uv pip install requirements_batch_face_detection.txt
```

Otherwise, please follow the section below to re-create the virtual environment.

## Install `torch`
``` bash
uv pip install torch==2.7.0 torchvision torchaudio
```

## Install `facenet-pytorch`
``` bash
uv pip install facenet-pytorch==2.5.3
```

## Install Utility Packages

``` bash
uv pip install pandas pyarrow pyspark Pillow 
```
Install other utility packages as needed.

## Confirm Installation
Test from command line:
``` bash
python -c "import torch; from facenet_pytorch import MTCNN; print('torch:', torch.__version__); print('facenet_pytorch loaded')"
```

# Camera-Trap Animal Detection

``` bash
uv venv batch_camera_trap
source batch_camera_trap/bin/activate  # Linux OS

# Verify python version
which python
```
If you have:
- Linux OS
- Python 3.12
- CUDA 12.4.1

Consider install dependencies from `requirements_batch_camera_trap.txt` 
``` bash 
uv pip install requirements_batch_camera_trap.txt
```

Otherwise, please follow the section below to re-create the virtual environment.

## Install `torch`
``` bash
uv pip install torch==2.7.0 torchvision torchaudio
```

## Install `PytorchWildlife`

During processing, `PytorchWildlife` has yet to have an efficient batch inference API for Megadetector to process large volume images using multiple GPUs. We opened a [PR](https://github.com/microsoft/CameraTraps/pull/577) to address this challenge. The feature has been merged into main in the upstream repository but is subject to [ongoing modifications](https://github.com/microsoft/CameraTraps/pull/577#pullrequestreview-2792951692). This [pre-release](https://github.com/NetZissou/CameraTraps/releases/tag/pw_v1.2.0-TOL) ensures that the processing pipeline for TreeOfLife Data remains reproducible during this period of `PytorchWildlife` active development.

``` bash
uv pip install git+https://github.com/NetZissou/CameraTraps.git@pw_v1.2.0-TOL
```
## Install Utility Packages

``` bash
uv pip install pandas pyarrow pyspark Pillow 
```
Install other utility packages as needed.

# All-in-One

``` bash
uv venv batch_processing
source batch_processing/bin/activate  # Linux OS

# Verify python version
which python
```
If you have:
- Linux OS
- Python 3.12
- CUDA 12.4.1

Consider install dependencies from `requirements_batch_processing.txt` 
``` bash 
uv pip install requirements_batch_processing.txt
```

Otherwise, please follow the section previous sections to create an all-in-one processing environment. 
