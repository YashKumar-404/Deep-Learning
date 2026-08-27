from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "train"
TEST_DIR = ROOT / "test"
LABELS_CSV = ROOT / "train_labels.csv"
CKPT_PATH = ROOT / "best_model.pth"
SUBMISSION_PATH = ROOT / "submission.csv"

MODEL_NAME = "resnet18"        # resnet18 | resnet34 | efficientnet_b0
IMG_SIZE = 96
VAL_SPLIT = 0.1
SEED = 42

EPOCHS = 5
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
