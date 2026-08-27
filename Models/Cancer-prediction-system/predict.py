import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from dataset import PCamDataset, build_transforms
from model import build_model


def main():
    if not config.CKPT_PATH.exists():
        sys.exit(f"No checkpoint at {config.CKPT_PATH}. Train first: python train.py")
    if not config.TEST_DIR.exists() or not any(config.TEST_DIR.glob("*.tif")):
        sys.exit(f"No test images in {config.TEST_DIR}")

    ids = [p.stem for p in config.TEST_DIR.glob("*.tif")]
    ds = PCamDataset(pd.DataFrame({"id": ids}), config.TEST_DIR, build_transforms(False))
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS, pin_memory=True)

    ckpt = torch.load(config.CKPT_PATH, map_location=config.DEVICE)
    model = build_model(ckpt["model_name"]).to(config.DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    out_ids, out_probs = [], []
    with torch.no_grad():
        for imgs, batch_ids in loader:
            imgs = imgs.to(config.DEVICE, non_blocking=True)
            with torch.autocast(device_type=config.DEVICE.type, enabled=config.USE_AMP):
                logits = model(imgs)
            out_probs.extend(torch.sigmoid(logits).float().cpu().numpy().ravel().tolist())
            out_ids.extend(batch_ids)

    pd.DataFrame({"id": out_ids, "label": out_probs}).to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Wrote {config.SUBMISSION_PATH} ({len(out_ids)} rows)")


if __name__ == "__main__":
    main()
