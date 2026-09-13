import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import config


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running = 0.0
    for imgs, labels in tqdm(loader, desc="train", leave=False):
        imgs = imgs.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True).unsqueeze(1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=config.DEVICE.type, enabled=config.USE_AMP):
            loss = criterion(model(imgs), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item() * imgs.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running = 0.0
    probs, targets = [], []
    for imgs, labels in tqdm(loader, desc="val", leave=False):
        imgs = imgs.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True).unsqueeze(1)
        with torch.autocast(device_type=config.DEVICE.type, enabled=config.USE_AMP):
            out = model(imgs)
            running += criterion(out, labels).item() * imgs.size(0)
        probs.append(torch.sigmoid(out).float().cpu())
        targets.append(labels.cpu())
    probs = torch.cat(probs).numpy().ravel()
    targets = torch.cat(targets).numpy().ravel()
    auc = roc_auc_score(targets, probs)
    acc = ((probs > 0.5).astype(float) == targets).mean()
    return running / len(loader.dataset), auc, acc
