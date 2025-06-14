# SAMFeat (Local features detection and description)

Implementation of IEEE TIP paper "Segment Anything Model is a Good Teacher for Local Feature Learning" (https://ieeexplore.ieee.org/document/10945548/).

Keywords: Local features detection and description; local descriptors; image matching; Segment Aything Model.

To do：
- [x] Evaluation code and Trained model for SAMFeat
- [x] Training code

# Requirement
```
conda env create -f environment.yml,
```

# Quick Evaluation
HPatches Image Matching Benchmark

1. Download trained SAMFeat model:

```cd ckpt```

Use the link https://drive.google.com/file/d/1NTRGZ2aJnT59_6b-n_SFY33jOwwxCJOM/view?usp=drive_link to download our trained model checkpoint from Google Drive. Place it under the ```ckpt``` folder.

2. Download HPatches benchmark: 

```cd evaluation_hpatch/hpatches_sequences``` then ```bash download.sh```

3. configure evaluation file:

Edit ```SAMFeat_eva.yaml``` file located in the ```configs``` folder

4. Extract local descriptors：
```
cd evaluation_hpatch
python export.py --top-k 10000 --tag SAMFeat --output_root output_path --config PATH_TO_SAMFeat_eva.yaml
```
This will extract descriptors and place it under the output folder

5. Evaluation
```
python get_score.py
```
This will print out the MMA score from threshold 1-to-10 and output a Pdf MMA Curve

# Training from Scratch
```
python train.py --gpus 1 --configs PATH_TO/configs/SAMFeat_train.yaml --indicator NAME
```
