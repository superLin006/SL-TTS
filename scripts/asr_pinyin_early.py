import json
import re
from pathlib import Path
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

ROOT = Path("/mnt/data/itc_tts")
WAV = ROOT / "eval/pinyin_pron_224"
mapping = json.loads((ROOT / "pron_scan/pron_mapping.json").read_text(encoding="utf-8"))
model = AutoModel(model="/mnt/data/jhc/Projects/voice/SenseVoiceSmall", device="cuda:0", disable_update=True)
def norm(s): return re.sub(r"[\s，。！？、；：“”‘’（）《》…—,.!?;:\"'()\[\]<>-]", "", s)
bad = []
for r in mapping:
    wav = WAV / f"{r['index']}.wav"
    res = model.generate(input=str(wav), language="zh", use_itn=False)
    text = rich_transcription_postprocess(res[0]["text"]) if res else ""
    if r["word"] not in norm(text):
        bad.append((r["id"], r["word"], text))
print("early_missing", len(bad))
for x in bad:
    print("??", x[0], x[1], "|", x[2])
