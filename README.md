# Task 3 - Mini9 Image Classification (CutMix + MixUp)

This project trains a 9-class image classifier with a custom ResNet-style model in PyTorch.
The main implementation is in `CutMix.ipynb` and includes data preprocessing, model definition, training, checkpointing, and evaluation.
![Leaderboard Result](./leaderboard.jpeg)
## Main Files

- `CutMix.ipynb`: full training and validation pipeline
- `model.py`: competition inference wrapper (`Model.predict`)
- `best_model.pth`: best checkpoint (saved from training loop)
- `EDA.ipynb`: exploratory analysis notebook

## Target Classes

- airplane
- automobile
- bird
- cat
- deer
- dog
- horse
- ship
- truck

## Data Layout Expected by Notebook

The notebook expects class folders under split directories:

```text
train/
  airplane/*.jpg
  automobile/*.jpg
  ...
val/
  airplane/*.jpg
  automobile/*.jpg
  ...
```

`Mini9Dataset` scans `*.jpg` files from each class folder and maps class names to indices `[0..8]`.

## Preprocessing Pipeline

### 1. Dataset loading

- Image read: `PIL.Image.open(...).convert("RGB")`
- Class mapping is fixed in this order:
  `['airplane','automobile','bird','cat','deer','dog','horse','ship','truck']`

### 2. Normalization statistics

- `mean = [0.52461963, 0.55828897, 0.58782098]`
- `std  = [0.18552486, 0.18374406, 0.19512308]`

### 3. Train transforms

```python
transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

### 4. Validation transforms

```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

### 5. Dataloader setup

- `batch_size = 128`
- train loader: `shuffle=True`, `num_workers=0`, `pin_memory=False`
- val loader: `shuffle=False`, `num_workers=0`, `pin_memory=False`

## Modeling Details

Model: `Mini9ResNet` (custom lightweight ResNet-style CNN)

### Block design (`ResidualBlock`)

- Main branch:
  - `3x3 Conv -> BatchNorm -> ReLU`
  - `3x3 Conv -> BatchNorm`
  - optional `Dropout2d` (stage-dependent)
- Shortcut branch:
  - identity if shape matches
  - otherwise projection with `1x1 Conv (stride)` + `BatchNorm`
- Output:
  - `ReLU(main + shortcut)`

### Network stages

- Stem: `Conv(3->64, 3x3)` + `BN` + `ReLU`
- Layer1: `64 -> 64`, 2 residual blocks, stride 1, dropout 0.1
- Layer2: `64 -> 128`, 2 residual blocks, stride 2, dropout 0.2
- Layer3: `128 -> 256`, 2 residual blocks, stride 2, dropout 0.3
- Layer4: `256 -> 512`, 2 residual blocks, stride 2, dropout 0.4
- Head: `AdaptiveAvgPool2d(1x1)` -> flatten -> `Linear(512 -> 9)`

### Initialization

- Conv: Kaiming normal
- BatchNorm: weight = 1, bias = 0
- Linear: normal(mean=0, std=0.01), bias = 0

Parameter count printed in notebook: `11,173,449`.

## Training Strategy

### Device

- Uses `mps` if available (Apple Silicon), otherwise CPU

### Hyperparameters

- `EPOCHS = 150`
- `LR = 0.1`
- `WEIGHT_DECAY = 5e-4`
- `LABEL_SMOOTHING = 0.1`
- `MIXUP_ALPHA = 1.0`
- `CUTMIX_PROB = 0.5`
- Early stopping patience: `25`

### Loss

Custom `LabelSmoothingCrossEntropy(smoothing=0.1)`.

### MixUp/CutMix policy

Applied online inside each training batch:

- Sample `lam ~ Beta(alpha, alpha)` with `alpha=1.0`
- With probability `0.5`: apply CutMix
- Otherwise: apply MixUp
- Combined loss:
  `L = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)`

For training accuracy with mixed labels, a weighted accuracy is used:
`lam * acc(pred, y_a) + (1-lam) * acc(pred, y_b)`.

### Optimizer and scheduler

- Optimizer: `SGD(momentum=0.9, nesterov=True, lr=0.1, weight_decay=5e-4)`
- LR scheduler: `CosineAnnealingLR(T_max=150, eta_min=1e-6)`
- Scheduler stepped once per epoch

### Checkpointing and early stopping

- Best model is selected by highest validation accuracy
- Saved as `best_model.pth`
- If no validation improvement for 25 epochs, training stops early

## Evaluation in Notebook

The notebook includes:

- overall validation accuracy
- confusion matrix
- per-class precision / recall / F1 report

Shown run result in notebook output: `Overall Accuracy: 92.93%` on validation split.

## Inference (`model.py`)

`model.py` defines a `Model` class used for evaluation platforms:

- loads `Mini9ResNet` + `best_model.pth`
- applies the same normalization statistics
- predicts labels in class-name format

`predict` expects input shape: `(N, 32, 32, 3)`.

## Requirements

```bash
pip install torch torchvision numpy pandas scikit-learn pillow matplotlib
```

## Quick Run

1. Open `CutMix.ipynb`.
2. Ensure `train/` and `val/` folders exist with class subfolders.
3. Run notebook cells in order.
4. Best checkpoint is written to `best_model.pth`.
