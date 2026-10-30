import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gen_dialogues as g
from matrix import ENGLISH_TERMS, NUMBERS, POLY

ROOT = Path(__file__).parent
RAW = ROOT / "long_sft" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

SYSTEM = "你是中文口语语料设计师，只输出严格 JSON，不要解释。"

PROMPT = """请生成 {n} 条自然中文长句，用于语音合成模型的长句训练。每条都是一个人独立说出的一句话，不要对话、不要说话人标签。

硬性要求：
1. 每条 40–70 个汉字，必须包含 2–4 个分句，用逗号、顿号或分号连接；整条只能有一个句末标点（。！？），不要写成两句话。
2. 口语自然、信息具体，像真人在讲一件小事；不要新闻联播腔，不要“首先…其次…最后”，不要“随着…的发展”，不要排比，不要空泛套话。
3. 句式多样：叙述、解释、提醒、转述、比较、计划、抱怨、建议都要有；同一批不要连续用相同开头。
4. 数字、英文、单位保持原样书写，例如 2026年10月26日、3点半、128元、2.3%、138 0000 0000、CA1832、1806室、USB-C、iPhone 15。
5. 内容健康，不涉及真实个人隐私；电话用 138 0000 0000，邮箱用 example.com。
6. 类别：{cat}；主题：{theme}
7. {extra}
8. 不要重复你前面生成过的句子；不要用下面这些套话：
   - 今天天气不错，我们出去走走吧。
   - 这个多少钱？可以刷卡吗？
   - 请把会议改到明天下午三点。
9. 不要出现“首先、其次、最后、综上所述、总而言之、随着”这些词。

只输出 JSON：
{{"sentences":[{{"text":"..."}}]}}
"""

BAD = [
    r"作为(一个)?(AI|人工智能)", r"以下是", r"首先", r"其次", r"最后",
    r"随着.*(发展|进步)", r"综上所述", r"总而言之", r"需要注意的是",
    r"(?:自杀|自残|色情|裸聊|毒品|赌博|枪支|炸弹|诈骗教程)",
    r"[a-z]{3,}\d{4,}",
    r"(?<![A-Za-z0-9])[A-Z]{2}-[A-Z0-9]{5,}(?![A-Za-z0-9])",
]
MOBILE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
EMAIL = re.compile(r"[\w.+-]+@[\w.-]+")
LATIN = re.compile(r"[A-Za-z][A-Za-z0-9.+#-]*")
HAN = re.compile(r"[\u4e00-\u9fff]")


def build_specs():
    general = [
        ("日常叙述", "菜市场买菜和做饭"), ("日常叙述", "家务和家电维修"),
        ("日常叙述", "邻居和社区琐事"), ("日常叙述", "家人和孩子"),
        ("日常叙述", "宠物和天气"), ("工作沟通", "周会汇报和进度"),
        ("工作沟通", "需求评审和排期"), ("工作沟通", "客户拜访和跟进"),
        ("工作沟通", "请假调休和交接"), ("解释说明", "操作步骤和注意事项"),
        ("解释说明", "原因解释和对比选择"), ("解释说明", "规则和流程说明"),
        ("服务场景", "餐厅点菜和外卖"), ("服务场景", "酒店入住和退改"),
        ("服务场景", "快递物流和售后"), ("旅行", "问路换乘和景点"),
        ("旅行", "行李安检和住宿"), ("教育", "家长老师和作业"),
        ("健康", "挂号问诊和用药"), ("情感叙事", "回忆和安慰"),
    ]
    specs = []
    for cat, theme in general:
        specs.append({"cat": cat, "theme": theme, "kind": "general"})
    for i in range(10):
        types = [NUMBERS[(i * 3 + k) % len(NUMBERS)] for k in range(3)]
        specs.append({"cat": "数字信息", "theme": "、".join(types), "kind": "numbers", "focus": types})
    for i in range(10):
        groups = [POLY[(i * 3 + k) % len(POLY)] for k in range(3)]
        specs.append({"cat": "多音字", "theme": "、".join(x[0] for x in groups), "kind": "poly", "focus": groups})
    for i in range(10):
        terms = [ENGLISH_TERMS[(i * 5 + k) % len(ENGLISH_TERMS)] for k in range(5)]
        specs.append({"cat": "中英混排", "theme": "手机电脑、App、网络或办公", "kind": "mixed", "focus": terms})
    return specs


def extra_text(spec):
    if spec["kind"] == "numbers":
        return f"本批至少 30 条自然包含这些数字/信息类型中的至少一种：{'、'.join(spec['focus'])}；每条至少 45 个汉字、至少两个分句，不要只写短句，也不要每条都带数字。"
    if spec["kind"] == "poly":
        words = "；".join(f"{p[0]}：{p[1]}" for p in spec["focus"])
        return f"本批至少 30 条自然包含这些多音字词：{words}；每个词至少出现 4 次，不要生硬堆砌。"
    if spec["kind"] == "mixed":
        return f"本批至少 30 条自然包含这些英文词中的至少 2 个：{'、'.join(spec['focus'])}；英文嵌在中文里，不要整句英文。"
    return "每句都要有具体的地点、物品、时间或原因，不要空泛。"


def check_batch(texts, spec):
    issues = []
    if len(texts) < 30:
        issues.append(f"有效句子只有 {len(texts)} 条")
    if any(len(t) > 78 for t in texts):
        issues.append("有句子超过 75 字")
    if len(set(texts)) < len(texts) * 0.9:
        issues.append("批内重复句太多")
    joined = "\n".join(texts)
    if spec["kind"] == "numbers":
        pat = "|".join(g.NUMBER_PATTERNS[x] for x in spec["focus"])
        hit = sum(1 for t in texts if re.search(pat, t))
        if hit < 20:
            issues.append(f"数字命中只有 {hit} 条")
    if spec["kind"] == "poly":
        for word, hint, reading in spec["focus"]:
            words = [w for w in hint.split("、") if w]
            total = sum(joined.count(w) for w in words)
            if total < 4:
                issues.append(f"“{word}”只出现 {total} 次")
    if spec["kind"] == "mixed":
        latin_turns = [t for t in texts if LATIN.search(t)]
        tokens = [x for t in texts for x in LATIN.findall(t) if len(x) >= 2]
        if len(latin_turns) < 20 or len(set(x.lower() for x in tokens)) < 5:
            issues.append(f"英文覆盖不足：{len(latin_turns)} 句、{len(set(x.lower() for x in tokens))} 个不同词")
    return issues


def clean(text):
    text = str(text).strip().replace("\n", "")
    text = re.sub(r"\s+", "", text)
    parts = [p for p in re.split(r"(?<=[，；、])", text) if p.strip()]
    if len(parts) < 2:
        return None
    if len(text) < 35 or len(text) > 78:
        return None
    if len(HAN.findall(text)) < 28:
        return None
    if text[-1] not in "。！？":
        text += "。"
    if text.count("。") + text.count("！") + text.count("？") != 1:
        return None
    if any(re.search(p, text) for p in BAD):
        return None
    for m in MOBILE.findall(text):
        if m != "13800000000":
            return None
    for e in EMAIL.findall(text):
        if not e.lower().endswith("@example.com"):
            return None
    if text.count("……") > 1:
        return None
    return text


def run_one(spec, idx, avoid):
    path = RAW / f"spec_{idx:03d}.json"
    if path.exists():
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            if d.get("ok"):
                return idx, "skip", len(d["sentences"])
        except Exception:
            pass
    prompt = PROMPT.format(
        n=50, cat=spec["cat"], theme=spec["theme"], extra=extra_text(spec),
    ) + "\n最近用过的开头（避免重复）：" + "、".join(avoid)
    last = None
    for attempt in range(3):
        try:
            resp = g.call_api(prompt)
            data = json.loads(resp["choices"][0]["message"]["content"])
            texts = []
            for item in data.get("sentences", []):
                t = clean(item.get("text", ""))
                if t:
                    texts.append(t)
            issues = check_batch(texts, spec)
            if issues:
                last = "；".join(issues)
                prompt = prompt + "\n上一版的问题：" + last + "。请重写全部句子。"
                continue
            path.write_text(json.dumps({"ok": True, "spec": spec, "sentences": texts,
                                        "usage": resp.get("usage"), "ts": time.time()},
                                       ensure_ascii=False, indent=1), encoding="utf-8")
            return idx, "ok", len(texts)
        except Exception as e:
            last = str(e)
            time.sleep(2 + attempt * 2)
    path.write_text(json.dumps({"ok": False, "spec": spec, "error": last}, ensure_ascii=False), encoding="utf-8")
    return idx, f"fail: {last}", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args()
    specs = build_specs()
    idxs = [int(x) for x in args.only.split(",")] if args.only else list(range(len(specs)))
    avoid_pool = ["我昨天", "今天早上", "上周", "我跟你说", "其实", "那个", "我们", "这事儿"]
    rng = random.Random(29)
    done = total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, specs[i], i, rng.sample(avoid_pool, 4)): i for i in idxs}
        for fut in as_completed(futs):
            i, status, n = fut.result()
            done += 1
            total += n
            print(f"[{done}/{len(idxs)}] spec_{i:03d} {status} n={n} total={total}", flush=True)
    print("DONE", done, "sentences", total)


if __name__ == "__main__":
    main()
