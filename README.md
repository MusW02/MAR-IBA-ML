# Offroad Terrain Semantic Segmentation

## Overview
This project implements a **semantic segmentation** pipeline for offroad terrain imagery using **UNet++** with an **EfficientNet-B4** encoder backbone. The model is trained on the Offroad Segmentation Training Dataset to classify each pixel into one of **10 terrain classes**: Background, Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Logs, Rocks, Landscape, and Sky.

## Trained Model
The trained model weights are located at **`submission1/best_model_unetpp.pth`**. This file contains the best-performing UNet++ model state dictionary saved during training based on the lowest validation loss.

## Project Structure
```
└── submission1/
    ├── best_model_unetpp.pth         # ★ Trained model weights (UNet++ EfficientNet-B4)
    ├── checkpoint.pth                # Training checkpoint (includes optimizer state)
    ├── train-test.ipynb             # Training notebook (submission version)
    ├── test.ipynb                   # Test evaluation notebook (submission version)
    ├── final_test_report.txt         # Evaluation metrics on the public test set
    └── test_results_public_80/
        └── visualizations/           # Prediction overlay visualizations
```

## Model Architecture & Training Details
- **Architecture:** UNet++ (segmentation_models_pytorch)
- **Encoder:** EfficientNet-B4, pretrained on ImageNet
- **Input Size:** 512 × 512 pixels
- **Number of Classes:** 10
- **Loss Function:** Weighted combination of CrossEntropy (0.4) + Dice Loss (0.6)
- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-2)
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=3)
- **Mixed Precision:** Enabled via `torch.cuda.amp` (GradScaler + autocast)
- **Epochs:** 40
- **Batch Size:** 8
- **Augmentations:** HorizontalFlip, ShiftScaleRotate, ISONoise/GaussNoise, RandomBrightnessContrast, HueSaturationValue, RGBShift, CoarseDropout

## Class Mapping
| Class ID | Index | Class Name     |
|----------|-------|----------------|
| 100      | 0     | Background     |
| 200      | 1     | Trees          |
| 300      | 2     | Lush Bushes    |
| 500      | 3     | Dry Grass      |
| 550      | 4     | Dry Bushes     |
| 600      | 5     | Ground Clutter |
| 700      | 6     | Logs           |
| 800      | 7     | Rocks          |
| 7100     | 8     | Landscape      |
| 10000    | 9     | Sky            |

## Test Results (Public Test Set — 80 images)
- **Mean IoU:** 0.3431
- **Inference Speed:** 25.13 ms/image

| Class          | IoU    |
|----------------|--------|
| Background     | 0.4071 |
| Trees          | 0.0006 |
| Lush Bushes    | 0.4735 |
| Dry Grass      | 0.2764 |
| Dry Bushes     | 0.0000 |
| Ground Clutter | 0.0000 |
| Logs           | 0.0000 |
| Rocks          | 0.0210 |
| Landscape      | 0.6714 |
| Sky            | 0.9764 |

## Requirements
- Python 3.8+
- PyTorch (with CUDA support recommended)
- segmentation-models-pytorch
- albumentations
- timm
- OpenCV (`cv2`)
- NumPy, Matplotlib, pandas, tqdm

Install dependencies:
```bash
pip install segmentation-models-pytorch albumentations timm opencv-python matplotlib pandas tqdm
```

## Usage
### Training
Run `submission1/train-test1.ipynb` (or `train-test.ipynb`) in Google Colab or locally. Training supports **checkpoint resumption** — if a `checkpoint.pth` exists, training resumes from the last saved epoch.

### Inference / Evaluation
Run `submission1/test.ipynb` to evaluate the trained model on the test set. It loads weights from `submission1/best_model_unetpp.pth`, computes per-class IoU, mean IoU, and saves prediction visualizations to `test_results_public_80/visualizations/`.