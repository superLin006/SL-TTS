import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import FakeQuantize, MovingAverageMinMaxObserver, PerChannelMinMaxObserver
from piper_train.vits.lightning import VitsModel

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--dataset-dir", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()


class QATConv1d(nn.Conv1d):
    def __init__(self, conv):
        super().__init__(conv.in_channels, conv.out_channels, conv.kernel_size,
                         conv.stride, conv.padding, conv.dilation, conv.groups,
                         conv.bias is not None, conv.padding_mode)
        self.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            self.bias.data.copy_(conv.bias.data)
        self.weight_fq = FakeQuantize(observer=PerChannelMinMaxObserver, quant_min=0,
                                      quant_max=255, ch_axis=0)
        self.act_fq = FakeQuantize(observer=MovingAverageMinMaxObserver, quant_min=0,
                                   quant_max=255)

    def forward(self, x):
        w = self.weight_fq(self.weight)
        y = F.conv1d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return self.act_fq(y)


def wrap(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv1d):
            setattr(module, name, QATConv1d(child))
        else:
            wrap(child)


def unwrap(module):
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
            unwrap(child)


ckpt = torch.load(args.checkpoint, map_location="cpu")
h = dict(ckpt["hyper_parameters"])
h["dataset"] = [str(Path(args.dataset_dir) / "dataset.jsonl")]
model = VitsModel(**h)
wrap(model.model_g.enc_p)
wrap(model.model_g.dp)
missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
print("load missing", len(missing), "unexpected", len(unexpected), flush=True)
if missing:
    print("missing sample", missing[:5], flush=True)
unwrap(model.model_g.enc_p)
unwrap(model.model_g.dp)
clean = dict(ckpt)
clean["state_dict"] = model.state_dict()
torch.save(clean, args.out)
print("saved", args.out, flush=True)
