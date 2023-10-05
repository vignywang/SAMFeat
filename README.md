# SAMFeat (Local features detection and description)

Implementation of "Segment Anything Model is a Good Teacher for Local Feature Learning".

The original paper can be found here: https://arxiv.org/pdf/2309.16992.pdf

Keywords: Local features detection and description; local descriptors; image matching; Segment Aything Model.

<!---
# Requirement
```
pip install -r requirement.txt,
```

# Quick start
HPatches Image Matching Benchmark

1.The trained model: The model checkpoint is located in folder ```models```

2.Extract local descriptors：
```
python export.py --top-k 10000 --tag SAMFeat --output_root output_path --config SAMFeat_eva.yaml
```
3.Evaluation
```
python get_score.py
```

# File Description
Folder ```hpatch_related``` contains the pytorch dataset class for HPatches
Folder ```models``` contains the PyTorch implementation of SAMFeat
```export.py``` and ```get_score.py``` are used to extract descriptors and evaluate HPatches
```requirements.txt``` is the environment installation reference 
-->
