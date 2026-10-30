import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import FakeQuantize, MovingAverageMinMaxObserver, PerChannelMinMaxObserver
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from piper_train.vits.lightning import VitsModel

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--dataset-dir", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--max-phoneme-ids", type=int, default=600)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--limit-batches", type=float, default=0)
args = ap.parse_args()


class QATConv1d(nn.Conv1d):
    def __init__(self, conv: nn.Conv1d):
        super().__init__(conv.in_channels, conv.out_channels, conv.kernel_size,
                         conv.stride, conv.padding, conv.dilation, conv.groups,
                         conv.bias is not None, conv.padding_mode)
        self.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            self.bias.data.copy_(conv.bias.data)
        self.weight_fq = FakeQuantize(
            observer=PerChannelMinMaxObserver, quant_min=0, quant_max=255, ch_axis=0)
        self.act_fq = FakeQuantize(
            observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255)

    def forward(self, x):
        w = self.weight_fq(self.weight)
        y = F.conv1d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return self.act_fq(y)


def wrap_convs(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv1d):
            setattr(module, name, QATConv1d(child))
        else:
            wrap_convs(child)


def unwrap_convs(module):
    for name, child in list(module.named_children()):
        if isinstance(child, QATConv1d):
            conv = nn.Conv1d(child.in_channels, child.out_channels, child.kernel_size,
                             child.stride, child.padding, child.dilation, child.groups,
                             child.bias is not None, child.padding_mode)
            conv.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                conv.bias.data.copy_(child.bias.data)
            setattr(module, name, conv)
        else:
            unwrap_convs(child)


ckpt = torch.load(args.checkpoint, map_location="cpu")
h = dict(ckpt["hyper_parameters"])
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
print("load missing", len(missing), "unexpected", len(unexpected), flush=True)
wrap_convs(model.model_g.enc_p)
wrap_convs(model.model_g.dp)
print("QAT convs", sum(1 for _ in model.model_g.enc_p.modules() if isinstance(_, QATConv1d)),
      sum(1 for _ in model.model_g.dp.modules() if isinstance(_, QATConv1d)), flush=True)

callbacks = [ModelCheckpoint(every_n_epochs=1, save_top_k=-1)]
trainer = Trainer(
    accelerator="gpu", devices=1, max_epochs=args.epochs,
    default_root_dir=args.out_dir, callbacks=callbacks,
    precision=32, num_sanity_val_steps=0, check_val_every_n_epoch=5,
    limit_train_batches=args.limit_batches if args.limit_batches > 0 else 1.0,
    enable_progress_bar=True,
)
trainer.fit(model)

unwrap_convs(model.model_g.enc_p)
unwrap_convs(model.model_g.dp)
clean = dict(ckpt)
clean["state_dict"] = model.state_dict()
out = Path(args.out_dir) / "qat_clean.ckpt"
torch.save(clean, out)
print("saved", out, flush=True)
