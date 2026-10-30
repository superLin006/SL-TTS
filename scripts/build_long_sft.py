import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
RAW = ROOT / "long_sft" / "raw"
OUT = ROOT / "long_sft_2000.jsonl"
REPORT = ROOT / "long_sft_report.md"

PUNCT = re.compile(r"[\s，。！？、；：“”‘’（）《》…—,.!?;:\"'()\[\]<>-]")


def norm(t):
    return PUNCT.sub("", t)


def shingles(t, n=8):
    t = norm(t)
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))}


def jac(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def main():
    cands = []
    for path in sorted(RAW.glob("spec_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        if not d.get("ok"):
            continue
        kind = d["spec"]["kind"]
        cat = d["spec"]["cat"]
        for t in d["sentences"]:
            cands.append({"kind": kind, "cat": cat, "text": t})

    refs = []
    for name in ["corpus_v2.jsonl", "../pilot_corpus_200.jsonl", "long_eval_50.jsonl"]:
        p = ROOT / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                refs.append(shingles(json.loads(line)["text"]))
    ref_index = defaultdict(list)
    for i, sh in enumerate(refs):
        for g in list(sh)[:20]:
            if len(ref_index[g]) < 300:
                ref_index[g].append(i)

    seen = set()
    kept = []
    removed = Counter()
    local_index = defaultdict(list)
    for c in cands:
        key = norm(c["text"])
        if key in seen:
            removed["exact_dup"] += 1
            continue
        sh = shingles(c["text"])
        near = False
        cand_ids = set()
        for g in list(sh)[:12]:
            cand_ids.update(local_index.get(g, ())[:200])
        for j in cand_ids:
            if jac(sh, shingles(kept[j]["text"])) >= 0.85:
                near = True
                break
        if near:
            removed["near_dup"] += 1
            continue
        ref_hit = False
        cand_refs = set()
        for g in list(sh)[:12]:
            cand_refs.update(ref_index.get(g, ())[:200])
        for j in cand_refs:
            if jac(sh, refs[j]) >= 0.5:
                ref_hit = True
                break
        if ref_hit:
            removed["overlap"] += 1
            continue
        idx = len(kept)
        kept.append(c)
        seen.add(key)
        for g in sh:
            if len(local_index[g]) < 300:
                local_index[g].append(idx)

    quotas = {"general": 800, "numbers": 400, "poly": 400, "mixed": 400}
    by_kind = defaultdict(list)
    for c in kept:
        by_kind[c["kind"]].append(c)
    rng = random.Random(31)
    selected = []
    for kind, quota in quotas.items():
        pool = by_kind.get(kind, [])
        rng.shuffle(pool)
        take = min(quota, len(pool))
        selected.extend(pool[:take])
        print(kind, "pool", len(pool), "take", take)
    if len(selected) < 2000:
        used = {id(x) for x in selected}
        rest = [c for c in kept if id(c) not in used]
        rng.shuffle(rest)
        selected.extend(rest[:2000 - len(selected)])
    selected = selected[:2000]
    selected.sort(key=lambda x: x["cat"])

    with OUT.open("w", encoding="utf-8") as f:
        for i, c in enumerate(selected, 1):
            f.write(json.dumps({"id": f"S{i:04d}", "cat": c["cat"], "kind": c["kind"], "text": c["text"]},
                               ensure_ascii=False) + "\n")

    lens = [len(c["text"]) for c in selected]
    cats = Counter(c["cat"] for c in selected)
    kinds = Counter(c["kind"] for c in selected)
    report = [
        "# 长句 SFT 语料\n",
        f"- 候选：{len(cands)}；去重/查重后：{len(kept)}；入选：{len(selected)}",
        f"- 过滤：{dict(removed)}",
        f"- 长度：{min(lens)}–{max(lens)}，平均 {sum(lens)/len(lens):.1f}",
        f"- 类型：{dict(kinds)}",
        f"- 类别：{dict(cats)}",
    ]
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
