import json
from pathlib import Path
from pinyin_frontend import PinyinFrontend

ROOT = Path("/mnt/data/itc_tts")
pf = PinyinFrontend("/mnt/data/itc_tts/pinyin/tokens.txt", "/mnt/data/itc_tts/pinyin/lexicon.txt")
tests = [
    ("还书", "我明天去图书馆还书。"),
    ("系鞋带", "你鞋带松了，赶紧蹲下系鞋带吧。"),
    ("种花", "我妈特别喜欢在阳台上种花。"),
    ("饕餮", "面对满桌好菜，他像饕餮一样吃个不停。"),
    ("蹙眉", "他听到这个消息，不由得蹙眉想了好一会儿。"),
    ("数一下", "你把椅子数一下，看够不够坐。"),
    ("银行", "我下午得去趟银行，把工资卡上的钱取出来。"),
]
for w, s in tests:
    ids, missing, toks = pf.ids(s)
    print(w, "| tokens", " ".join(toks), "| ids", len(ids), "missing", missing)
rows = [json.loads(x) for x in (ROOT / "pron_scan/pron_sentences.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
bad = 0
for r in rows:
    ids, missing, toks = pf.ids(r["text"])
    if missing or not ids:
        bad += 1
        print("BAD", r["id"], r["word"], missing, toks[:30])
print("pron rows", len(rows), "bad", bad)
