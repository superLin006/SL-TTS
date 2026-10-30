import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
RAW = ROOT / "raw"
PILOT = ROOT.parent / "pilot_corpus_200.jsonl"
OUT = ROOT / "corpus_v2.jsonl"
REPORT = ROOT / "report.md"

HAN = re.compile(r"[\u4e00-\u9fff]")
LATIN = re.compile(r"[A-Za-z][A-Za-z0-9.+#-]*")
PUNCT = re.compile(r"[\s，。！？、；：“”‘’（）《》…—,.!?;:\"'()\[\]<>-]")
MOBILE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
ID18 = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
EMAIL = re.compile(r"[\w.+-]+@[\w.-]+")

BAD_PATTERNS = [
    r"作为(一个)?(AI|人工智能)", r"抱歉[，,]?\s*我(无法|不能)", r"以下是", r"对话(如下|内容)",
    r"场景[:：]", r"^\s*[AB][:：]", r"首先.*其次.*最后", r"随着.*(发展|进步)",
    r"综上所述", r"总而言之", r"需要注意的是", r"您好[，,]?\s*请问有什么可以帮",
    r"(?:自杀|自残|色情|裸聊|毒品|赌博|枪支|炸弹|诈骗教程)",
    r"[a-z]{3,}\d{4,}",
    r"(?<![A-Za-z0-9])[A-Z]{2}-[A-Z0-9]{5,}(?![A-Za-z0-9])",
]

NUMBER_TYPES = {
    "年份": r"(?:19|20)\d{2}\s*年",
    "具体日期": r"\d{1,2}\s*月\s*\d{1,2}\s*[日号]",
    "星期": r"(?:周|星期)[一二三四五六日天]",
    "时刻": r"\d{1,2}\s*[点:：]\s*\d{0,2}",
    "时长": r"\d+\s*(?:分钟|小时|钟头|秒)",
    "金额": r"\d+(?:\.\d+)?\s*(?:元|块|万|千|块钱)",
    "百分比": r"\d+(?:\.\d+)?\s*%|百分之",
    "折扣": r"\d+(?:\.\d+)?\s*折|满\s*\d+\s*减\s*\d+",
    "电话号码": r"1[3-9]\d[\s-]?\d{4}[\s-]?\d{4}|400[\s-]?\d{3}[\s-]?\d{4}",
    "订单号": r"(?:订单号|单号)[^。！？]{0,12}\d{6,}",
    "房间号": r"\d{3,4}\s*(?:室|房|房间)",
    "楼层": r"(?:\d{1,3}|[一二三四五六七八九十百]+)\s*(?:楼|层)",
    "航班号": r"[A-Z]{2}\s?\d{3,4}",
    "车次": r"[GDCZTK]\s?\d{1,4}",
    "身份证后四位": r"后四位[^。！？]{0,8}\d{4}",
    "重量": r"\d+(?:\.\d+)?\s*(?:克|千克|公斤|斤|两|吨)",
    "尺寸": r"\d+(?:\.\d+)?\s*(?:厘米|毫米|米|寸|英寸)",
    "温度": r"(?:零下)?\d+(?:\.\d+)?\s*(?:度|℃)",
    "速度": r"\d+(?:\.\d+)?\s*(?:公里|千米|迈|Mbps|兆)",
    "里程": r"\d+(?:\.\d+)?\s*(?:公里|千米)",
    "比分": r"\d+\s*[比:：]\s*\d+",
    "评分": r"\d+(?:\.\d+)?\s*分",
    "比例": r"\d+\s*[比:：]\s*\d+|\d+成",
    "人数": r"\d+\s*(?:个人|人|位)",
    "数量": r"\d+\s*(?:个|件|箱|斤|份|台|套|双|本)",
    "年龄": r"\d+\s*(?:岁|周岁)",
    "编号": r"(?:编号|号)[^。！？]{0,10}\d{2,}",
    "邮编": r"\b\d{6}\b",
    "利率": r"\d+(?:\.\d+)?\s*%|利率",
    "汇率": r"\d+(?:\.\d+)?\s*(?:汇率|比|兑)|汇率",
    "电量": r"\d+\s*%|电量|没电",
    "流量": r"\d+\s*(?:G|GB|M|MB|个G)",
    "内存大小": r"\d+\s*(?:G|GB|TB|MB)",
}

POLY_WORDS = {
    "行": ["银行", "行长", "行业", "行规", "自行车", "步行", "行不行", "行车", "行情"],
    "重": ["重要", "重新", "重复", "重庆", "重量", "重口味", "体重", "重心", "重组"],
    "还": ["还是", "还有", "还了", "归还", "还款", "还书", "还给", "偿还"],
    "乐": ["音乐", "乐器", "快乐", "乐意", "乐园", "音乐会", "欢乐", "乐曲"],
    "长": ["长江", "很长", "长大", "增长", "行长", "长度", "长期", "长城", "长发", "延长", "擅长", "校长"],
    "得": ["觉得", "记得", "得到", "你得", "得分", "心得", "值得"],
    "背": ["背包", "背在", "背景", "背诵", "后背", "背面", "背心", "背单词"],
    "调": ["调整", "空调", "调查", "调换", "调节", "调试", "声调", "曲调"],
    "差": ["差一点", "差别", "出差", "参差", "差距", "差错", "时差", "温差"],
    "着": ["拿着", "着急", "穿着", "着火", "接着", "跟着", "睡着", "着装"],
    "为": ["为了", "因为", "成为", "认为", "以为", "作为", "行为", "为何"],
    "发": ["发现", "头发", "发票", "发型", "出发", "发生", "发送", "发货", "理发"],
    "数": ["数字", "数一下", "数量", "数据", "数学", "多数", "少数", "数数"],
    "空": ["空调", "有空", "空间", "填空", "空气", "天空", "空白", "抽空"],
    "干": ["干净", "干活", "干燥", "干部", "干脆", "饼干", "树干", "能干"],
    "种": ["种子", "种类", "种花", "种植", "种菜", "品种", "播种"],
    "处": ["处理", "到处", "相处", "办事处", "好处", "处方", "四处", "处境"],
    "传": ["传统", "上传", "传记", "传递", "传说", "宣传", "流传"],
    "系": ["关系", "系鞋带", "系统", "联系", "系列", "系上"],
    "角": ["角度", "角色", "角落", "三角形", "主角", "牛角"],
    "率": ["效率", "利率", "率领", "概率", "频率", "率先", "坦率"],
    "便": ["方便", "便宜", "便利", "便当", "随便"],
    "参": ["参加", "人参", "参差", "参考", "参与", "参观", "海参"],
    "薄": ["薄饼", "薄弱", "薄片", "薄被", "单薄", "稀薄"],
    "露": ["露天", "露面", "露出", "露台", "露水", "揭露", "暴露", "流露"],
}


def norm_key(text):
    return PUNCT.sub("", text)


def shingles(text, n=6):
    t = norm_key(text)
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_pilot():
    rows = [json.loads(x) for x in PILOT.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [(r["text"], shingles(r["text"], 8)) for r in rows]


def clean_text(text):
    text = text.strip()
    text = re.sub(r"—+", "，", text)
    text = re.sub(r"\.{3,}", "……", text)
    text = re.sub(r"……{2,}", "……", text)
    text = text.strip("，、；： ")
    if text and text[-1] not in "。！？":
        text += "。"
    return text


def check_text(text, kind):
    reasons = []
    if len(text) < 5 or len(text) > 70:
        reasons.append("length")
    if not HAN.search(text):
        reasons.append("no_chinese")
    if any(re.search(p, text) for p in BAD_PATTERNS):
        reasons.append("bad_pattern")
    for m in MOBILE.findall(text):
        if m.replace(" ", "") != "13800000000":
            reasons.append("real_mobile")
    if ID18.search(text):
        reasons.append("id18")
    for e in EMAIL.findall(text):
        if not e.lower().endswith("@example.com"):
            reasons.append("real_email")
    if text.count("……") > 1 or ".." in text:
        reasons.append("ellipsis")
    if len(LATIN.findall(text)) > 6 and kind != "mixed":
        reasons.append("too_much_latin")
    return reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-ratio", type=float, default=0.02)
    args = ap.parse_args()

    batches = []
    for path in sorted(RAW.glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("ok") and d.get("dialogues"):
            batches.append(d)
    print(f"batches={len(batches)}")

    pilot = load_pilot()
    rows = []
    seen = {}
    shingle_index = defaultdict(list)
    removed = Counter()
    kept_dialogues = 0

    for batch in batches:
        spec = batch["spec"]
        for di, dlg in enumerate(batch["dialogues"]):
            kept = []
            for ti, turn in enumerate(dlg["turns"]):
                text = turn["text"].strip()
                text = clean_text(text)
                reasons = check_text(text, spec["kind"])
                if reasons:
                    removed.update(reasons)
                    continue
                key = norm_key(text)
                if key in seen:
                    removed["exact_dup"] += 1
                    continue
                sh = shingles(text, 6)
                dup = False
                cand = set()
                for g in list(sh)[:12]:
                    cand.update(shingle_index.get(g, ())[:200])
                for j in cand:
                    if jaccard(sh, shingles(rows[j]["text"], 6)) >= 0.8:
                        dup = True
                        break
                if dup:
                    removed["near_dup"] += 1
                    continue
                pilot_dup = False
                for ptext, psh in pilot:
                    if key == norm_key(ptext) or jaccard(shingles(text, 8), psh) >= 0.6:
                        pilot_dup = True
                        break
                if pilot_dup:
                    removed["pilot_overlap"] += 1
                    continue
                idx = len(rows)
                rows.append({
                    "id": f"v{len(rows) + 1:05d}",
                    "kind": spec["kind"],
                    "scene": spec["scene"],
                    "focus": spec.get("focus_number"),
                    "poly_group": spec["poly"][0],
                    "dialogue": f"{spec['id']}_{di:02d}",
                    "spk": turn["spk"],
                    "text": text,
                    "chars": len(text),
                })
                seen[key] = idx
                for g in sh:
                    if len(shingle_index[g]) < 400:
                        shingle_index[g].append(idx)
                kept.append(idx)
            if len(kept) >= 3:
                kept_dialogues += 1

    # 全局短句上限 10%
    short = [r for r in rows if r["chars"] <= 4]
    if len(short) > len(rows) * 0.1:
        drop = {r["id"] for r in short[int(len(rows) * 0.1):]}
        rows = [r for r in rows if r["id"] not in drop]
        removed["short_trim"] += len(drop)

    # 问句/感叹句限流，避免陈述句语调被带偏
    rng = random.Random(7)

    def cap_ratio(pred, ratio, name):
        nonlocal rows
        idxs = [i for i, r in enumerate(rows) if pred(r)]
        target = int(len(rows) * ratio)
        if len(idxs) > target:
            drop = set(rng.sample(idxs, len(idxs) - target))
            rows = [r for i, r in enumerate(rows) if i not in drop]
            removed[name] += len(drop)

    cap_ratio(lambda r: r["text"].rstrip().endswith("？"), 0.22, "question_trim")
    cap_ratio(lambda r: r["text"].rstrip().endswith("！"), 0.10, "exclaim_trim")

    def cap_match(pattern, cap, name):
        nonlocal rows
        idxs = [i for i, r in enumerate(rows) if re.search(pattern, r["text"])]
        if len(idxs) > cap:
            drop = set(rng.sample(idxs, len(idxs) - cap))
            rows = [r for i, r in enumerate(rows) if i not in drop]
            removed[name] += len(drop)

    cap_match(r"138\s*0000\s*0000", 10, "phone_trim")
    cap_match(r"example\.com", 8, "email_trim")
    cap_match(r"1806", 40, "room_trim")

    # 分层 train/val
    by_kind = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r)
    for kind, items in by_kind.items():
        n_val = max(1, int(len(items) * args.val_ratio))
        for r in items[:n_val]:
            r["split"] = "val"
        for r in items[n_val:]:
            r["split"] = "train"

    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 覆盖度报告
    texts = [r["text"] for r in rows]
    joined = "\n".join(texts)
    num_hits = {k: sum(1 for t in texts if re.search(p, t)) for k, p in NUMBER_TYPES.items()}
    poly_hits = {}
    for group, words in POLY_WORDS.items():
        counts = {w: joined.count(w) for w in words}
        poly_hits[group] = (sum(counts.values()), {w: c for w, c in counts.items() if c})
    latin_tokens = Counter(t for t in texts for t in LATIN.findall(t) if len(t) >= 2)
    lens = [r["chars"] for r in rows]
    hours = sum(lens) / 4.4 / 3600
    kinds = Counter(r["kind"] for r in rows)
    scenes = Counter(r["scene"] for r in rows)

    report = []
    report.append("# corpus v2 报告\n")
    report.append(f"- 批次：{len(batches)}，有效对话：{kept_dialogues}")
    report.append(f"- 句子：{len(rows)}（train {sum(1 for r in rows if r['split']=='train')} / val {sum(1 for r in rows if r['split']=='val')}）")
    report.append(f"- 总字数：{sum(lens)}，估算音频：{hours:.1f} 小时")
    report.append(f"- 平均长度：{sum(lens)/len(lens):.1f} 字；最短 {min(lens)}，最长 {max(lens)}")
    report.append(f"- 长度分布：≤12 {sum(1 for x in lens if x<=12)} / 13–28 {sum(1 for x in lens if 13<=x<=28)} / 29–55 {sum(1 for x in lens if 29<=x<=55)} / >55 {sum(1 for x in lens if x>55)}")
    report.append(f"- 类型：{dict(kinds)}")
    report.append(f"- 过滤：{dict(removed)}\n")
    focus_counts = Counter(r["focus"] for r in rows if r.get("focus"))
    report.append("## 数字/信息覆盖（专项句数 / 字面命中）\n")
    for k in NUMBER_TYPES:
        report.append(f"- {k}: {focus_counts[k]} / {num_hits[k]}")
    report.append("\n## 多音字覆盖（总次数 / 分项）\n")
    for k, (total, detail) in poly_hits.items():
        report.append(f"- {k}: {total} {detail}")
    report.append("\n## 英文高频词\n")
    for k, v in latin_tokens.most_common(40):
        report.append(f"- {k}: {v}")
    report.append("\n## 场景覆盖（前 40）\n")
    for k, v in scenes.most_common(40):
        report.append(f"- {k}: {v}")
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report[:20]))
    print(f"\nwrote {OUT} and {REPORT}")


if __name__ == "__main__":
    main()
