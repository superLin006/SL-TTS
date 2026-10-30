import json
import sys
from pathlib import Path

import jieba
from pypinyin import load_phrases_dict
from misaki.zh import ZHG2P

ROOT = Path("/mnt/data/itc_tts")
sys.path.insert(0, str(ROOT))
import frontend_zh

phrases = json.loads((ROOT / "pinyin/lexicon_phrases.json").read_text(encoding="utf-8"))
custom = json.loads((ROOT / "pinyin/custom_phrases.json").read_text(encoding="utf-8"))
g2p = ZHG2P(version="1.1")
for w in list(phrases) + list(custom):
    jieba.add_word(w, freq=100000)
load_phrases_dict(phrases)
load_phrases_dict(custom)

sets = [
    ROOT / "train/v2/corpus.jsonl",
    ROOT / "train/v2_long/long_sft_2000.jsonl",
]
bad = 0
total = 0
for corpus in sets:
    for line in corpus.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        text, _ = frontend_zh.normalize_text(r["text"])
        total += 1
        try:
            g2p(text)
        except Exception as e:
            bad += 1
            hits = [w for w in phrases if w in text and len(w) > 1]
            hits.sort(key=len, reverse=True)
            print("FAIL", r["id"], repr(e), "|", text, "| hits", hits[:3], flush=True)
            if bad >= 20:
                break
    if bad >= 20:
        break
print("total", total, "bad", bad)
