import argparse
import torch
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--sft", required=True)
ap.add_argument("--alpha", type=float, required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

base = torch.load(args.base, map_location="cpu")
sft = torch.load(args.sft, map_location="cpu")
out = dict(base)
sd = dict(base["state_dict"])
count = 0
for k, v in base["state_dict"].items():
    if k.startswith("model_g.") and k in sft["state_dict"]:
        a = sft["state_dict"][k]
        if v.shape == a.shape and v.is_floating_point():
            sd[k] = (args.alpha * v.float() + (1.0 - args.alpha) * a.float()).to(v.dtype)
            count += 1
out["state_dict"] = sd
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
torch.save(out, args.out)
print("saved", args.out, "averaged", count, "tensors", "alpha", args.alpha)
