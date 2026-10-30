import json
from pathlib import Path
from pinyin_frontend import PinyinFrontend

ROOT = Path("/mnt/data/itc_tts")
pf = PinyinFrontend(str(ROOT / "pinyin/tokens.txt"), str(ROOT / "pinyin/lexicon.txt"))
for name in ["pron_mapping.json", "student_mapping_long.json", "student_mapping_200.json"]:
    p = ROOT / "pron_scan" / name if name == "pron_mapping.json" else ROOT / "eval" / name
    rows = json.loads(p.read_text(encoding="utf-8"))
    out = ROOT / "eval" / (name.replace("_mapping", "_input").replace(".json", "_pinyin.jsonl"))
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            ids, missing, toks = pf.ids(r["text"])
            if missing:
                print("MISSING", name, r["id"], missing)
            f.write(json.dumps({"phoneme_ids": ids}, ensure_ascii=False) + "\n")
            n += 1
    print(name, "->", out.name, n)
