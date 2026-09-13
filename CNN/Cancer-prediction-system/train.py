import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import load_splits
from model import build_model
from engine import train_one_epoch, evaluate


def check_data():
    if not config.LABELS_CSV.exists():
        sys.exit(f"Missing labels file: {config.LABELS_CSV}")
    if not config.TRAIN_DIR.exists() or not any(config.TRAIN_DIR.glob("*.tif")):
        sys.exit(
            f"No training images in {config.TRAIN_DIR}\n"
            "Download the Kaggle 'Histopathologic Cancer Detection' train images and "
            "unzip them into that folder (files named <id>.tif). See README.md."
        )


def main():
    check_data()
    torch.manual_seed(config.SEED)

    train_ds, val_ds = load_splits()
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)

    model = build_model().to(config.DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP)

    print(f"Device: {config.DEVICE} | Model: {config.MODEL_NAME} | "
          f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    best_auc = 0.0
    for epoch in range(1, config.EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, auc, acc = evaluate(model, val_loader, criterion)
        scheduler.step()
        print(f"Epoch {epoch}/{config.EPOCHS} | train_loss {tr_loss:.4f} | "
              f"val_loss {val_loss:.4f} | AUC {auc:.4f} | acc {acc:.4f}")
        if auc > best_auc:
            best_auc = auc
            torch.save({"model_name": config.MODEL_NAME, "state_dict": model.state_dict(),
                        "val_auc": auc, "epoch": epoch}, config.CKPT_PATH)
            print(f"  saved {config.CKPT_PATH.name} (AUC {auc:.4f})")

    print(f"Best val AUC: {best_auc:.4f}")

 
if __name__ == "__main__":
    main()
