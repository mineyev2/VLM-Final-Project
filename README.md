# VLM Final Project

## PACE-ICE Usage
### 1) Connecting to VPN
You must be connected to VPN (even when on eduroam) in order to use PACE-ICE.

(**TODO**: Insert tutorial)

### 2) Connecting to PACE-ICE
```
ssh <GT_USERNAME>@login-ice.pace.gatech.edu
```

### 3) Uploading Folders/Files
There are 3 ways to upload:
1) `scp` command on terminal
2) onDemand website
3) Globus
4) (**TODO**: Can I upload through some URL?)

For big folders, I could only get Globus working. The tutorial is here: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0041890

NOTE: When setting up Global Connect Personal Setup, do NOT select "High Assurance". You need a special certificate for that to work on the server side.

### 4) GPU Usage
**TODO**

## OpenEMMA Installation
(Original repo: https://github.com/taco-group/OpenEMMA/tree/main)

### 1) Clone repo

1) **IMPORTANT**: Navigate to the `scratch` directory first!
2) Clone repo in the `scratch` directory

### 2) Setup Virtual Environment
```
conda create -n openemma python=3.8
conda activate openemma
```

### 3) Install Dependencies
```
cd <repo-folder>
conda install nvidia/label/cuda-12.4.0::cuda-toolkit
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 4) Download Dataset

1) Visit nuScenes and make an account: https://www.nuscenes.org/nuscenes#download
2) Navigate to Full Dataset v1.0 > Mini > US and download
3) Upload the downloaded `dataset` folder to repo root folder (`<HOME_DIR>/scratch/VLM-Final-Project/`) using Globus (instructions are described above)

**TODO (Roman)**: Update to use the whole dataset instead of just mini

### 5) Running the Code
#### Update HuggingFace Directory
Add the following line to the end of your `~/.bashrc` file (for model/datasets to be stored in scratch instead of local storage):
```
export HF_HOME=<HOME_DIR>/scratch/.cache/huggingface
```

Then, create a subfolder for this it (.cache/huggingface/hub does not exist by default)
```
cd <SCRATCH_DIR>
mkdir -p .cache/huggingface/hub
```

#### Running the Script
```
python main.py \
    --model-path [model] \
    --dataroot [dir-of-nuscnse-dataset] \
    --version [version-of-nuscnse-dataset] \
    --method openemma
```

- `--model-path`: I use qwen
- `--dataroot`: If you made a new folder according to Step 3, should not need this argument
- `--version`: Version of dataset (name you see in the zip file when you download. For me it's "v1.0-mini")


### Things we could try to fix:
- `pip install wrapt`
- If we want to use QWEN2.5-VL need to upgrade python version to 3.10 or something
    - transformers library not updated for new models for python 3.8