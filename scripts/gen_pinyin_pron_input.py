import json
from pathlib import Path
from pinyin_frontend import PinyinFrontend

ROOT = Path("/mnt/data/itc_tts")
pf = PinyinFrontend("/mnt/data/itc_tts/pinyin/tokens.txt", "/mnt/data/itc_tts/pinyin/lexicon.txt")
rows = json.loads((ROOT / "pron_scan/pron_mapping.json").read_text(encoding="utf-8"))
with (ROOT / "pron_scan/student_input_pron_pinyin.jsonl").open("w", encoding="utf-8") as f:
    for r in rows:
        ids, missing, toks = pf.ids(r["text"])
        if missing:
            print("MISSING", r["id"], missing)
        f.write(json.dumps({"phoneme_ids": ids}, ensure_ascii=False) + "\n")
print("rows", len(rows))
