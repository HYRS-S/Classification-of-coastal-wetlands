# README.md

```markdown
# MSSTANet: Multi-Scale Spatial-Temporal Attention Network for Hyperspectral Image Classification

This repository contains the official implementation of the MSSTANet model for hyperspectral image (HSI) classification, as described in our paper.

## Repository Structure

```

MSSTANet/
├── checkpoints/          # Saved model weights (ignored by git)
├── data/                 # HSI datasets (LHK_VN.tif, LHK_SW.tif)
├── trainTestSplit/       # Train/test split indices
├── HSIDataset.py         # Dataset loader
├── model.py              # MSSTANet architecture
├── train.py              # Training script
├── test.py               # Evaluation script
├── utils.py              # Training utilities
└── README.md             # This file

````

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended, but CPU is supported)

Install dependencies:

```bash
pip install -r requirements.txt
````

## Data Preparation

1. Place your hyperspectral images in the `data/` folder.
2. Expected files:

   * `data/{dataset}_VN.tif` (Visible-NIR bands)
   * `data/{dataset}_SW.tif` (Short-wave infrared bands)
3. The model concatenates both files along the band dimension.

Example for dataset `LHK`:

```
data/
├── LHK_VN.tif
└── LHK_SW.tif
```

## Usage

### Training

```bash
python train.py --dataset LHK --ratio 0.75 --epochs 100 --batch_size 256 --lr 1e-3
```

Arguments:

* `--dataset` : Name of the dataset (default: LHK)
* `--ratio`   : Training ratio (default: 0.75)
* `--epochs`  : Number of epochs (default: 100)
* `--batch_size` : Batch size (default: 256)
* `--lr`      : Learning rate (default: 1e-3)

### Testing

After training, test the model:

```bash
python test.py --dataset LHK --checkpoint checkpoints/best_model.pth
```


