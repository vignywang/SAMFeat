# SAMFeat (Local features detection and description)

Implementation of "Segment Anything Model is a Good Teacher for Local Feature Learning".

Keywords: Local features detection and description; local descriptors; image matching; Segment Aything Model.

To do：
- [x] Evaluation code and Trained model for SANFeat
- [ ] Training code (Coming soon)

# Requirement
```
conda env create -f environment.yml,
```

# Quick start
HPatches Image Matching Benchmark

1. ```cd ckpt``` and use the link to download our trained model checkpoint. Place it under the ```ckpt``` folder.

2. configure evaluation file ```SAMFeat_eva.yaml``` file located in the ```configs``` folder

3. Extract local descriptors：
```
cd evaluation_hpatch
python export.py --top-k 10000 --tag SAMFeat --output_root output_path --config PATH_TO_SAMFeat_eva.yaml
```
This will extract descriptors and place it under the output folder

4. Evaluation
```
python get_score.py
```
This will print out the MMA score from threshold 1-to-10 and output a Pdf MMA Curve
