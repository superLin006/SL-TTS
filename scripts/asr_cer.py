import json
import re
import statistics
import sys
from pathlib import Path

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

wav_dir = sys.argv[1]
mapping_path = sys.argv[2]
out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None
mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
model = AutoModel(model="/mnt/data/jhc/Projects/voice/SenseVoiceSmall", device="cuda:0", disable_update=True)


def norm(s):
    return re.sub(r"[\s，。！？、；：“”‘’（）《》…—,.!?;:\"'()\[\]<>-]", "", s)


def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


rows = []
for i, r in enumerate(mapping):
    res = model.generate(input=f"{wav_dir}/{i}.wav", language="zh", use_itn=False)
    text = rich_transcription_postprocess(res[0]["text"]) if res else ""
    ref = norm(r["text"])
    hyp = norm(text)
    err = lev(ref, hyp)
    rows.append({"i": i, "id": r.get("id", i), "ref": ref, "asr": hyp,
                 "cer": round(err / max(1, len(ref)), 4), "err": err})
    if i % 25 == 0:
        print("progress", i, flush=True)

cers = [x["cer"] for x in rows]
exact = sum(1 for x in rows if x["cer"] == 0)
print("SUMMARY n", len(rows),
      "mean_cer", round(statistics.mean(cers), 4),
      "median_cer", round(statistics.median(cers), 4),
      "exact", exact,
      "total_err", sum(x["err"] for x in rows),
      "total_chars", sum(len(x["ref"]) for x in rows), flush=True)
if out_path:
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
