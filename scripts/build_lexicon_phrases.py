import json
from pathlib import Path

ROOT = Path("/mnt/data/itc_tts/pinyin")
tokens = {}
for line in (ROOT / "tokens.txt").read_text(encoding="utf-8").splitlines():
    if line.strip():
        sym, idx = line.rsplit(" ", 1)
        tokens[sym] = int(idx)

phrases = {}
for line in (ROOT / "lexicon.txt").read_text(encoding="utf-8").splitlines():
    parts = line.strip().split()
    if len(parts) < 2:
        continue
    word, toks = parts[0], parts[1:]
    if len(word) < 2 or len(word) > 8 or not all("\u4e00" <= c <= "\u9fff" for c in word):
        continue
    syls = []
    cur = []
    for t in toks:
        if t == "_":
            if cur:
                syls.append("".join(cur))
                cur = []
        elif t != "Ø":
            cur.append(t)
    if cur:
        syls.append("".join(cur))
    if len(syls) != len(word):
        continue
    phrases[word] = [[s] for s in syls]

(ROOT / "lexicon_phrases.json").write_text(json.dumps(phrases, ensure_ascii=False), encoding="utf-8")
print("tokens", len(tokens), "phrases", len(phrases))
for w in ["还书", "系鞋带", "种花", "银行", "行长", "重庆", "调查", "长大", "音乐", "背包", "头发", "有空", "干活", "处理", "角色", "率领", "便宜", "人参", "薄饼", "露面"]:
    print(w, phrases.get(w))
