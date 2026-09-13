# Histopathologic Cancer Detection — CNN

Binary classifier for the [Kaggle Histopathologic Cancer Detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection) dataset (PatchCamelyon). Each sample is a 96×96 RGB tissue patch; label `1` means the center 32×32 region contains tumor tissue. Built with transfer learning (ImageNet-pretrained CNN) in PyTorch.

## Data

Only `train_labels.csv` (220,025 rows, ~60/40 negative/positive) ships in this folder. The image patches (~7 GB) are **not** included — download them from Kaggle:

```bash
kaggle competitions download -c histopathologic-cancer-detection
```

Unzip so the folders sit next to the scripts:

```
Cancer-prediction-system/
├── train_labels.csv
├── train/            <id>.tif  (labeled, for training)
└── test/             <id>.tif  (unlabeled, for submission)
```

## Setup

Install the CUDA build of PyTorch for your GPU, then the rest:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

Stratified 90/10 split, mixed-precision, cosine LR schedule. Best-AUC checkpoint is saved to `best_model.pth`. Tune epochs/batch size/model in `config.py` (`MODEL_NAME`: `resnet18` | `resnet34` | `efficientnet_b0`).

## Predict

```bash
python predict.py
```

Runs the checkpoint over `test/` and writes `submission.csv` (`id,label` with `label` = tumor probability), ready for Kaggle upload.

## Files

| File | Role |
|------|------|
| `config.py` | paths + hyperparameters |
| `dataset.py` | `.tif` loading, augmentation, stratified split |
| `model.py` | pretrained backbone with a 1-logit head |
| `engine.py` | train / eval loops (loss, ROC-AUC, accuracy) |
| `train.py` | training entry point |
| `predict.py` | inference → `submission.csv` |

## Notes

- Metric is **ROC-AUC** (the competition metric), reported per epoch alongside accuracy.
- Windows: `num_workers > 0` needs the `if __name__ == "__main__"` guard (already in place).
- CPU works but is slow on 220k images; a GPU is expected. Set `NUM_WORKERS = 0` to debug.
