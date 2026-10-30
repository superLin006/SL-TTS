import json
from collections import Counter
from pathlib import Path

from piper_phonemize import phonemize_espeak, phoneme_ids_espeak
import frontend_zh

ROOT = Path("/mnt/data/itc_tts")
CORPUS = Path("/mnt/data/itc_tts/train/v2_long/long_sft_2000.jsonl")
OUT = Path("/mnt/data/itc_tts/train/v2_long")

rows = [json.loads(x) for x in CORPUS.read_text(encoding="utf-8").splitlines() if x.strip()]
records = []
rejected = Counter()
for i, row in enumerate(rows):
    normalized, notes = frontend_zh.normalize_text(row["text"])
    phonemes = [p for sent in phonemize_espeak(normalized, "cmn") for p in sent]
    missing = Counter()
    ids = phoneme_ids_espeak(phonemes, missing_phonemes=missing)
    n = len(ids)
    if n < 330:
        rejected["short"] += 1
        continue
    if n > 800:
        rejected["long"] += 1
        continue
    records.append({"index": i, "id": row["id"], "cat": row.get("cat"), "raw": row["text"],
                    "text": normalized, "phoneme_ids": ids, "missing": dict(missing), "notes": notes})

with (OUT / "sft_input.jsonl").open("w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps({"phoneme_ids": r["phoneme_ids"]}, ensure_ascii=False) + "\n")
with (OUT / "sft_mapping.json").open("w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=1)
lens = sorted(len(r["phoneme_ids"]) for r in records)
print("rows", len(rows), "kept", len(records), "rejected", dict(rejected), "max", max(lens),
      "p90", lens[int(len(lens) * 0.9)], "min", min(lens),
      "missing", {r["id"]: r["missing"] for r in records if r["missing"]})
