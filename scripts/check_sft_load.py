import torch
from piper_train.vits.lightning import VitsModel

ckpt_path = "/mnt/data/itc_tts/train/runs/medium_v2/snapshots/epoch=699.ckpt"
ckpt = torch.load(ckpt_path, map_location="cpu")
h = dict(ckpt["hyper_parameters"])
h.update({"dataset": ["/mnt/data/itc_tts/train/v2_long/piper_dataset/dataset.jsonl"],
          "max_phoneme_ids": 800, "batch_size": 16, "learning_rate": 5e-5,
          "num_workers": 1, "validation_split": 0.02, "num_test_examples": 0})
m = VitsModel(**h)
res = m.load_state_dict(ckpt["state_dict"], strict=False)
print("missing", len(res.missing_keys), res.missing_keys[:8])
print("unexpected", len(res.unexpected_keys), res.unexpected_keys[:8])
k = "model_g.enc_p.emb.weight"
print("same_tensor", torch.equal(m.state_dict()[k], ckpt["state_dict"][k]))
print("num_params", sum(p.numel() for p in m.parameters()))
