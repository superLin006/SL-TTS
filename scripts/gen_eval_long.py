import json
from collections import Counter
from pathlib import Path

from piper_phonemize import phonemize_espeak, phoneme_ids_espeak
import frontend_zh

ROOT = Path("/mnt/data/itc_tts")
CORPUS = ROOT / "eval/long_eval_50.jsonl"
OUT = ROOT / "eval"

rows = [json.loads(x) for x in CORPUS.read_text(encoding="utf-8").splitlines() if x.strip()]
records = []
for i, row in enumerate(rows):
    normalized, notes = frontend_zh.normalize_text(row["text"])
    phonemes = [p for sent in phonemize_espeak(normalized, "cmn") for p in sent]
    missing = Counter()
    ids = phoneme_ids_espeak(phonemes, missing_phonemes=missing)
    records.append({"index": i, "id": row["id"], "cat": row["cat"], "raw": row["text"],
                    "text": normalized, "phoneme_ids": ids, "missing": dict(missing), "notes": notes})

with (OUT / "student_input_long.jsonl").open("w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps({"phoneme_ids": r["phoneme_ids"]}, ensure_ascii=False) + "\n")
with (OUT / "student_mapping_long.json").open("w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=1)
print("rows", len(records), "empty", sum(1 for r in records if not r["phoneme_ids"]),
      "max_ph", max(len(r["phoneme_ids"]) for r in records),
      "missing", {r["id"]: r["missing"] for r in records if r["missing"]})
