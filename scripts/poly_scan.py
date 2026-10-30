import json
import re
import sys
from pathlib import Path

import jieba
from kokoro import KPipeline
from pypinyin import load_phrases_dict

sys.path.insert(0, "/mnt/data/itc_tts")
sys.path.insert(0, "/mnt/data/itc_tts/pinyin")
import frontend_zh
from pinyin_frontend import PinyinFrontend
from misaki.zh_frontend import ZH_MAP

ROOT = Path("/mnt/data/itc_tts")
REV = {}
for k, v in ZH_MAP.items():
    if len(v) == 1 and all("\u3105" <= c <= "\u312f" or c in "压言阳要阴应用又外万为王文瓮我元月云十" for c in v):
        REV[v] = k


def bop_syllable(s):
    m = re.match(r"(.+?)([1-5])$", s)
    if not m:
        return s
    base, tone = m.groups()
    out = []
    for ch in base:
        value = REV.get(ch, ch)
        if value in ("ii", "iii"):
            value = "i"
        out.append(value)
    return "".join(out) + ("" if tone == "5" else tone)


def teacher_chars(norm):
    _, tokens = pipe.g2p.frontend(norm)
    chars = []
    for tk in tokens:
        if not tk.text or not all("\u4e00" <= c <= "\u9fff" for c in tk.text):
            continue
        syls = re.findall(r"[^1-5]+[1-5]", tk.phonemes or "")
        if len(syls) != len(tk.text):
            return None
        chars.extend((c, bop_syllable(s)) for c, s in zip(tk.text, syls))
    return chars


def app_chars(norm):
    _, _, toks = pf.ids(norm)
    syls = []
    cur = []
    for t in toks:
        if t in ("^", "$", "_"):
            if t == "_" and cur:
                syls.append("".join(cur))
                cur = []
            continue
        if t == "Ø":
            continue
        if re.fullmatch(r"[1-5]", t):
            cur.append(t)
            continue
        if re.fullmatch(r"[A-Za-z]+", t):
            cur.append(t)
            continue
        # punctuation
        if cur:
            syls.append("".join(cur))
            cur = []
    if cur:
        syls.append("".join(cur))
    chinese = [c for c in norm if "\u4e00" <= c <= "\u9fff"]
    if len(syls) != len(chinese):
        return None
    return list(zip(chinese, syls))


def base(s):
    return re.sub(r"[1-5]$", "", s).replace("ü", "v")


def tone(s):
    m = re.search(r"([1-5])$", s)
    return m.group(1) if m else ""


pipe = KPipeline(lang_code="z", repo_id="hexgrad/Kokoro-82M-v1.1-zh", model=False)
frontend_zh.install_polyphone_dict()
phrases = json.loads((ROOT / "pinyin/lexicon_phrases.json").read_text(encoding="utf-8"))
custom = json.loads((ROOT / "pinyin/custom_phrases.json").read_text(encoding="utf-8"))
for w in list(phrases) + list(custom):
    jieba.add_word(w, freq=100000)
load_phrases_dict(phrases)
load_phrases_dict(custom)
pf = PinyinFrontend(str(ROOT / "pinyin/tokens.txt"), str(ROOT / "pinyin/lexicon.txt"))
input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "pron_scan/pron_mapping.json"
mapping = [json.loads(x) for x in input_path.read_text(encoding="utf-8").splitlines() if x.strip()]
rows = []
for r in mapping:
    norm, _ = frontend_zh.normalize_text(r["text"])
    tchars = teacher_chars(norm)
    achars = app_chars(norm)
    if tchars is None or achars is None:
        rows.append({"id": r["id"], "word": r["word"], "error": "align"})
        continue
    chars = [c for c, _ in tchars]
    tsyl = [s for _, s in tchars]
    asyl = [s for _, s in achars]
    word = r["word"]
    start = "".join(chars).find(word)
    if start < 0:
        rows.append({"id": r["id"], "word": word, "error": "word-not-found"})
        continue
    idx = list(range(start, start + len(word)))
    reading = r["reading"].split()
    t = [tsyl[i] for i in idx]
    a = [asyl[i] for i in idx]
    tbad = [base(x) for x in t] != [base(x) for x in reading]
    abad = [base(x) for x in a] != [base(x) for x in reading]
    ttone = [tone(x) for x in t] != [tone(x) for x in a]
    rows.append({"id": r["id"], "word": word, "reading": r["reading"],
                 "teacher": " ".join(t), "app": " ".join(a),
                 "base_mismatch": tbad or abad, "tone_diff": ttone})
print(json.dumps(rows, ensure_ascii=False, indent=1))
