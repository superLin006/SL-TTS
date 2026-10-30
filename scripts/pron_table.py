import json
from pathlib import Path
from pypinyin import lazy_pinyin, Style
from piper_phonemize import phonemize_espeak
import frontend_zh

ROOT = Path("/mnt/data/itc_tts")
frontend_zh.install_polyphone_dict()
mapping = json.loads((ROOT / "pron_scan/pron_mapping.json").read_text(encoding="utf-8"))
out = []
for r in mapping:
    w = r["word"]
    tone3 = " ".join(lazy_pinyin(w, style=Style.TONE3, neutral_tone_with_five=True))
    espeak = " ".join(p for sent in phonemize_espeak(w, "cmn") for p in sent)
    out.append({"id": r["id"], "word": w, "tone3": tone3, "espeak": espeak, "text": r["text"]})
(ROOT / "pron_scan/pron_table.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
for r in out:
    print(r["id"], r["word"], "|", r["tone3"], "|", r["espeak"])
