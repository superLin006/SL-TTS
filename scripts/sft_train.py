import argparse
import json
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from piper_train.vits.lightning import VitsModel

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--dataset-dir", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--max-phoneme-ids", type=int, default=800)
ap.add_argument("--workers", type=int, default=4)
args = ap.parse_args()

ckpt = torch.load(args.checkpoint, map_location="cpu")
h = dict(ckpt["hyper_parameters"])
print("old hparams", {k: h.get(k) for k in ["dataset", "max_phoneme_ids", "batch_size", "learning_rate"]}, flush=True)
h.update({
    "dataset": [str(Path(args.dataset_dir) / "dataset.jsonl")],
    "max_phoneme_ids": args.max_phoneme_ids,
    "batch_size": args.batch_size,
    "learning_rate": args.lr,
    "num_workers": args.workers,
    "validation_split": 0.02,
    "num_test_examples": 0,
})
model = VitsModel(**h)
missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
print("load_state missing", len(missing), "unexpected", len(unexpected), flush=True)
print("new hparams", {k: model.hparams.get(k) for k in ["dataset", "max_phoneme_ids", "batch_size", "learning_rate"]}, flush=True)

trainer = Trainer(
    accelerator="gpu", devices=1, max_epochs=args.epochs,
    default_root_dir=args.out_dir,
    callbacks=[ModelCheckpoint(every_n_epochs=10, save_top_k=-1)],
    precision=32, num_sanity_val_steps=0, check_val_every_n_epoch=10,
    enable_progress_bar=True,
)
trainer.fit(model)
