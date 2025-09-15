# VLM Final Project

## OpenEMMA Installation
(Original repo: https://github.com/taco-group/OpenEMMA/tree/main)

### 1) Setup Env
```
conda create -n openemma python=3.8
conda activate openemma
```

### 2) Install Dependencies
```
cd <repo-folder>
conda install nvidia/label/cuda-12.4.0::cuda-toolkit
pip install -r requirements.txt
```

### 3) Download Dataset
Folder should look like this (TODO: Check this, I just copied from nuscenes-devkit for now)
```
./datasets/NuScenes
    samples	-	Sensor data for keyframes (annotated images).
    sweeps  -   Sensor data for intermediate frames (unannotated images).
    v1.0-*	-	JSON tables that include all the meta data and annotations. Each split (train, val, test, mini) is provided in a separate folder.
```

### 4) Running Script
```
python main.py \
    --model-path qwen \
    --dataroot [dir-of-nuscnse-dataset] \
    --version [version-of-nuscnse-dataset] \
    --method openemma
```

- `--dataroot`: If you made a new folder according to Step 3, should not need this argument
- `--version`: Version of dataset (name you see in the zip file when you download. For me it's "v1.0-mini")


### Implementation things to fix:
- `pip install wrapt`
- If we want to use QWEN2.5-VL need to upgrade python version to 3.10 or something
    - transformers library not updated for new models for python 3.8
- 