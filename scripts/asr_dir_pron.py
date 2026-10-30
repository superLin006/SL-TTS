import json
import re
import sys

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

d = sys.argv[1]
rows = json.load(open("/mnt/data/itc_tts/pron_scan/pron_mapping.json", encoding="utf-8"))
model = AutoModel(model="/mnt/data/jhc/Projects/voice/SenseVoiceSmall", device="cuda:0", disable_update=True)


def norm(s):
    return re.sub(r"[\s，。！？、；：“”‘’（）《》…—,.!?;:\"'()\[\]<>-]", "", s)


bad = []
for i, r in enumerate(rows):
    res = model.generate(input=f"{d}/{i}.wav", language="zh", use_itn=False)
    text = rich_transcription_postprocess(res[0]["text"]) if res else ""
    if r["word"] not in norm(text):
        bad.append((i, r["id"], r["word"], text))
print("missing", len(bad))
for x in bad:
    print("??", x[0], x[1], x[2], "|", x[3])
