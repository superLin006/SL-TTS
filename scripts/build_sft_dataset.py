import json
import random
import os
from collections import Counter
from pathlib import Path

import soundfile as sf

ROOT = Path("/mnt/data/itc_tts")
SFT = ROOT / "train/v2_long"
TEACHER = SFT / "zf_001"
INPUT = SFT / "input"
WAV = INPUT / "wav"
WAV.mkdir(parents=True, exist_ok=True)

mapping = json.loads((SFT / "sft_mapping.json").read_text(encoding="utf-8"))
teacher = {}
for line in (TEACHER / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        r = json.loads(line)
        teacher[r["id"]] = r

new = []
for r in mapping:
    t = teacher.get(r["id"])
    if not t:
        continue
    new.append({"id": r["id"], "text": r["text"], "wav": str(TEACHER / t["wav"]),
                "audio_s": t["audio_s"], "group": "long_new"})

v2 = [json.loads(x) for x in (ROOT / "train/v2/piper_dataset/dataset.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
long_old = [r for r in v2 if len(r["phoneme_ids"]) > 400]
short_pool = [r for r in v2 if len(r["phoneme_ids"]) <= 400]
lo = [r for r in short_pool if len(r["phoneme_ids"]) <= 200]
hi = [r for r in short_pool if len(r["phoneme_ids"]) > 200]
rng = random.Random(101)
replay = rng.sample(lo, min(500, len(lo))) + rng.sample(hi, min(500, len(hi)))

rows = []
for r in new:
    dst = WAV / f"{r['id']}.wav"
    if not dst.exists():
        os.symlink(r["wav"], dst)
    rows.append((f"wav/{r['id']}.wav", r["text"], r["audio_s"], r["group"]))
for i, r in enumerate(long_old):
    dst = WAV / f"V{i:05d}.wav"
    if not dst.exists():
        os.symlink(r["audio_path"], dst)
    rows.append((f"wav/V{i:05d}.wav", r["text"], None, "long_old"))
for i, r in enumerate(replay):
    dst = WAV / f"R{i:05d}.wav"
    if not dst.exists():
        os.symlink(r["audio_path"], dst)
    rows.append((f"wav/R{i:05d}.wav", r["text"], None, "replay"))

with (INPUT / "metadata.csv").open("w", encoding="utf-8") as f:
    for wav, text, _, _ in rows:
        f.write(f"{wav}|{text}\n")

# audio durations
total = 0.0
for wav, _, _, _ in rows:
    try:
        total += sf.info(str(INPUT / wav)).duration
    except Exception:
        pass

stats = {
    "total": len(rows),
    "long_new": sum(1 for x in rows if x[3] == "long_new"),
    "long_old": sum(1 for x in rows if x[3] == "long_old"),
    "replay": sum(1 for x in rows if x[3] == "replay"),
    "audio_hours_approx": round(total / 3600, 2),
}
(INPUT / "sft_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(stats, ensure_ascii=False, indent=2))
