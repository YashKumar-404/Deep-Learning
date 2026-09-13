import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torchvision.transforms as T

import config


def build_transforms(train):
    norm = T.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
    if train:
        # histology patches are orientation-invariant -> flips are safe, strong augs
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            norm,
        ])
    return T.Compose([T.ToTensor(), norm])


class PCamDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_labels = "label" in df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.img_dir / f"{row['id']}.tif").convert("RGB")
        img = self.transform(img)
        if self.has_labels:
            return img, float(row["label"])
        return img, row["id"]


def load_splits():
    df = pd.read_csv(config.LABELS_CSV)
    train_df, val_df = train_test_split(
        df, test_size=config.VAL_SPLIT, stratify=df["label"], random_state=config.SEED
    )
    train_ds = PCamDataset(train_df, config.TRAIN_DIR, build_transforms(True))
    val_ds = PCamDataset(val_df, config.TRAIN_DIR, build_transforms(False))
    return train_ds, val_ds
