import argparse
import json
import time
from pathlib import Path

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

ROOT = Path("/mnt/data/itc_tts")
MODEL = "/mnt/data/jhc/Projects/voice/SenseVoiceSmall"
OUT = ROOT / "pron_scan/asr_sensevoice.jsonl"

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="teacher,soup03,base699,sft19")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

mapping = json.loads((ROOT / "pron_scan/pron_mapping.json").read_text(encoding="utf-8"))
if args.limit:
    mapping = mapping[:args.limit]
models = args.models.split(",")
model = AutoModel(model=MODEL, device=args.device, disable_update=True)

with OUT.open("w", encoding="utf-8") as f:
    done = 0
    for name in models:
        for r in mapping:
            if name == "teacher":
                wav = ROOT / "pron_scan/zf_001/wav" / f"{r['id']}.wav"
            else:
                wav = ROOT / "eval" / f"pron_{name}" / f"{r['index']}.wav"
            if not wav.exists():
                continue
            t0 = time.perf_counter()
            res = model.generate(input=str(wav), language="zh", use_itn=False, batch_size_s=60)
            text = rich_transcription_postprocess(res[0]["text"]) if res else ""
            f.write(json.dumps({"model": name, "id": r["id"], "index": r["index"],
                                "word": r["word"], "target": r["text"], "asr": text,
                                "sec": round(time.perf_counter() - t0, 2)},
                               ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            if done % 20 == 0:
                print("done", done, name, r["id"], round(time.perf_counter() - t0, 2), flush=True)
print("DONE", done, flush=True)
