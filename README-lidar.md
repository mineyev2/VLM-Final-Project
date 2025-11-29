```
conda env create -f ./environment.yml
conda activate lidarclip2

module load cuda/11.8   #Using cuda 11.8 for the env on Pace Ice
nvcc --version   #For checking cuda version

mim install mmcv==2.1.0
mim install mmdet==3.3.0
mim install mmdet3d==1.4.0

pip install numpy==1.26.4
```