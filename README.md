# Optimal Input Selector for Few-Shot 3DGS

---------------------------------------------------

## Environmental Setups
We provide install method based on Conda package and environment management:
```bash
conda env create --file environment.yml
conda activate FSGS
```
**CUDA 11.7** is strongly recommended.

## Data Preparation
In data preparation step, we reconstruct the sparse view inputs using SfM using the camera poses provided by datasets. Next, we continue the dense stereo matching under COLMAP with the function `patch_match_stereo` and obtain the fused stereo point cloud from `stereo_fusion`. 

``` 
cd FSGS
mkdir dataset 
cd dataset

# download LLFF dataset
gdown https://drive.google.com/uc?id=11PhkBXZZNYTD2emdG1awALlhCnkq7aN-

# run colmap to obtain initial point clouds with limited viewpoints
python tools/colmap_llff.py

# download MipNeRF-360 dataset
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip -d mipnerf360 360_v2.zip

# run colmap on MipNeRF-360 dataset
python tools/colmap_360.py
``` 

We provide both the sparse and dense point cloud after we proprecess them. You may download them [through this link](https://drive.google.com/drive/folders/1lYqZLuowc84Dg1cyb8ey3_Kb-wvPjDHA?usp=sharing). We use dense point cloud during training but you can still try sparse point cloud on your own.

## Generating COLMAP for Random Train Set
Generate COLMAP on LLFF dataset with randomly selected 3 views (10 random samples)
You can modify the number of sampling iterations at the tools/colmap_llff.py (num_runs)
```
python tools/colmap_llff.py
```

## Random Train Set Test (Train-Rander-Evaluate)
Train FSGS on LLFF dataset with randomly selected 3 views (10 random samples)
You can modify the number of sampling iterations at the run.py (num_runs)
```
python run.py
```