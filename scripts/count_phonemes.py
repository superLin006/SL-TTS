import json
from collections import Counter

p = "/mnt/data/itc_tts/train/v2/piper_dataset/dataset.jsonl"
b = Counter()
ids_lens = []
ph_lens = []
for line in open(p, encoding="utf-8"):
    d = json.loads(line)
    n = len(d.get("phoneme_ids") or [])
    ids_lens.append(n)
    ph_lens.append(len(d.get("phonemes") or []))
    if n <= 400:
        b["<=400"] += 1
    elif n <= 550:
        b["401-550"] += 1
    elif n <= 800:
        b["551-800"] += 1
    else:
        b[">800"] += 1
ids_lens.sort()
print("total", len(ids_lens), "ids min", ids_lens[0], "max", ids_lens[-1], "p90", ids_lens[int(len(ids_lens) * 0.9)])
print("phonemes max", max(ph_lens))
print(dict(b))
