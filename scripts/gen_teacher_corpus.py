import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import jieba
from pypinyin import load_phrases_dict
from kokoro import KModel, KPipeline

import frontend_zh

ROOT = Path("/mnt/data/itc_tts")
MODEL = ROOT / "models/kokoro-v1.1-zh"

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--lexicon", default="/mnt/data/itc_tts/pinyin/lexicon_phrases.json")
ap.add_argument("--custom", default="/mnt/data/itc_tts/pinyin/custom_phrases.json")
args = ap.parse_args()
IN = Path(args.corpus)
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

rows = [json.loads(x) for x in IN.read_text(encoding="utf-8").splitlines() if x.strip()]
if args.limit:
    rows = rows[:args.limit]
print("sentences", len(rows), flush=True)
t0 = time.perf_counter()
model = KModel(repo_id="hexgrad/Kokoro-82M-v1.1-zh", config=str(MODEL / "config.json"),
               model=str(MODEL / "kokoro-v1_1-zh.pth")).to("cuda").eval()
voice = torch.load(MODEL / "voices/zf_001.pt", map_location="cpu", weights_only=True)
pipe = KPipeline(lang_code="z", repo_id="hexgrad/Kokoro-82M-v1.1-zh", model=model)
frontend_zh.install_polyphone_dict()
phrases = json.loads(Path(args.lexicon).read_text(encoding="utf-8"))
custom = json.loads(Path(args.custom).read_text(encoding="utf-8"))
for w in list(phrases) + list(custom):
    jieba.add_word(w, freq=100000)
load_phrases_dict(phrases)
load_phrases_dict(custom)
print("load_s", round(time.perf_counter() - t0, 2), flush=True)

records = []
wav_dir = OUT / "wav"
wav_dir.mkdir(exist_ok=True)
skipped = 0
latin = 0
for i, row in enumerate(rows, 1):
    normalized, notes = frontend_zh.normalize_text(row["text"])
    if not normalized or any("a" <= ch.lower() <= "z" for ch in normalized):
        skipped += 1
        if normalized:
            latin += 1
            print("LATIN_LEFT", row["id"], normalized[:80], flush=True)
        continue
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    chunks = [audio for _, _, audio in pipe(normalized, voice=voice, speed=1.0)]
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    dur = len(audio) / 24000
    if dur < 0.4 or dur > 40 or not np.isfinite(audio).all():
        skipped += 1
        continue
    wav = wav_dir / f"{row['id']}.wav"
    sf.write(str(wav), audio, 24000)
    records.append({"id": row["id"], "cat": row.get("cat"), "raw": row["text"], "text": normalized,
                    "wav": str(wav.relative_to(OUT)).replace("\\", "/"), "audio_s": round(dur, 3),
                    "total_s": round(total, 3), "rtf": round(total / dur, 4) if dur else None,
                    "notes": notes})
    if i % 200 == 0 or i == len(rows):
        print(i, "/", len(rows), "kept", len(records), "skipped", skipped, "latin", latin, flush=True)

with (OUT / "manifest.jsonl").open("w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
summary = {"sentences": len(records), "skipped": skipped, "latin_left": latin,
           "audio_seconds": round(sum(r["audio_s"] for r in records), 1),
           "audio_hours": round(sum(r["audio_s"] for r in records) / 3600, 2),
           "rtf_median": round(statistics.median(r["rtf"] for r in records), 4) if records else None}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
