import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from matrix import BAD_EXAMPLES, make_spec

ROOT = Path(__file__).parent
RAW = ROOT / "raw"
RAW.mkdir(exist_ok=True)
KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip() or (ROOT.parent / ".deepseek_key").read_text(encoding="utf-8").strip()
URL = "https://api.deepseek.com/v1/chat/completions"

NUMBER_PATTERNS = {
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

LATIN = re.compile(r"[A-Za-z][A-Za-z0-9.+#-]*")
HAN = re.compile(r"[\u4e00-\u9fff]")

SYSTEM = "你是中文口语语料设计师，为轻量级语音合成模型编写真实、自然、多样的双人对话。你只输出严格 JSON，不要任何解释。"

PROMPT = """请生成 {n} 段真实、自然的双人对话，供语音合成模型训练使用。这些对话之后会被拆成单句朗读，所以每一句都必须自然、口语化、可独立朗读。

本批设定：
- 场景：{scene}
- 总体语气：{tone}
- 分 {ngroups} 组，每组 2–3 段，人物关系和目的不同：
{groups}

硬性要求：
1. 每段 4–6 轮，A/B 交替，每人 2–3 句。每句 5–55 个汉字；允许少量 2–4 字短应答，但最多占全部句子的 15%。本批至少 3 句超过 30 字。
2. 必须像真实口语：可以有“嗯、哎、那个、你看、说白了、行、好嘞、没事、要不、其实”等口语词，但不要每句都用；允许省略主语、倒装、追问、补充说明；不要教科书腔，不要每条都是完整主谓宾。
3. 句式必须多样：陈述、疑问、反问、祈使、感叹、确认、打断后改口都要出现；禁止连续使用相同开头；禁止排比；禁止“请…请…请…”。
4. 内容要具体：出现具体的地点、物品、时间、原因或数字，不要空泛。对话要围绕一个具体小事展开，有来有回，不要各说各话。同一批里不要反复用相同的句子骨架。
5. 不要出现舞台说明、括号动作、表情符号、Markdown、说话人姓名以外的标签。
6. 不要重复下面的反面示例，也不要重复你前面生成过的句子：
{blacklist}
7. 数字、英文、单位保持原样书写（后续由程序归一化）。例如：2026年10月26日下午3点半、138 0000 0000、2.3%、CA1832、1806室、USB-C、iPhone 15 Pro。
8. 内容健康，不涉及真实个人隐私；电话用 138 0000 0000 这类明显虚构号码；邮箱用 example.com。
9. 本批必须自然覆盖这些交际功能：{functions}。不要给每句贴标签，要让功能自然出现在对话里。
10. 长度分布：短句（≤12字）约 {s}%，中句（13–28字）约 {m}%，长句（29–55字）约 {l}%。
11. {number_req}
12. {poly_req}
{english_req}
14. {avoid_req}

只输出 JSON，格式：
{{"dialogues":[{{"scene":"一句话场景","turns":[{{"spk":"A","text":"..."}},{{"spk":"B","text":"..."}}]}}]}}
"""


def build_prompt(spec, avoid_openings):
    poly_word, poly_hint, poly_reading = spec["poly"]
    groups = "\n".join(
        f"  {i + 1}) 人物关系：{g['relation']}；目的：{g['goal']}"
        for i, g in enumerate(spec["groups"])
    )
    if spec["kind"] == "poly":
        poly_req = (
            f"本批是多音字专项练习。每一段对话至少自然出现 2 个含「{poly_word}」的词，"
            f"从这些词里选：{poly_hint}。必须把词放在能明确读音的上下文里（{poly_reading}），"
            "不要生硬堆砌，也不要只出现单独一个“行/重/还”字。"
        )
    else:
        poly_req = (
            f"本批至少 2 句自然包含含「{poly_word}」的词，可参考：{poly_hint}；"
            f"注意读音（{poly_reading}），不要生硬堆砌。"
        )
    english_req = (
        f"13. 本批至少自然出现这些英文词中的 4 个：{'、'.join(spec['english_pool'])}；"
        "至少 6 句含英文，英文要嵌在中文句子里，不要整句英文，也不要只出现单位符号。"
        if spec["english"]
        else "13. 本批不要出现英文单词或拉丁字母（单位符号除外），全部用中文表达。"
    )
    if spec["kind"] == "numbers":
        number_req = (
            f"本批是数字/信息专项。每一段对话都必须自然出现至少一个{spec['focus_number']}，"
            "至少 6 段要出现，而且不要都挤在开头；数字写法保持原样（如 2026年、3点半、128元、138 0000 0000、CA1832、1806室）。"
        )
    else:
        number_req = (
            f"本批必须自然出现这些数字/信息类型：{'、'.join(spec['numbers'])}。"
            "不要每句都带数字，至少 30% 的句子不含数字。"
        )
    avoid_req = (
        "不要用这些最近频繁出现的开头：" + "、".join(avoid_openings)
        if avoid_openings
        else "注意开头要多样，不要都用“我觉得”“那个”“您好”。"
    )
    s, m, l = spec["length"]
    blacklist = "\n".join(f"   - {x}" for x in BAD_EXAMPLES)
    return PROMPT.format(
        n=spec["n_dialogues"],
        scene=spec["scene"],
        tone=spec["tone"],
        ngroups=len(spec["groups"]),
        groups=groups,
        blacklist=blacklist,
        functions="、".join(spec["functions"]),
        s=s, m=m, l=l,
        number_req=number_req,
        poly_req=poly_req,
        english_req=english_req,
        avoid_req=avoid_req,
    )


def call_api(prompt, retries=6):
    body = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 1.15,
        "max_tokens": 6000,
        "response_format": {"type": "json_object"},
    }
    for attempt in range(retries):
        try:
            r = requests.post(
                URL,
                headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                timeout=180,
            )
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 3 * 2 ** attempt))
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(min(60, 3 * 2 ** attempt))
    raise RuntimeError("retries exhausted")


def parse_and_check(content, spec):
    data = json.loads(content)
    dialogues = data.get("dialogues")
    if not isinstance(dialogues, list):
        raise ValueError("missing dialogues")
    clean = []
    for d in dialogues:
        turns = d.get("turns")
        if not isinstance(turns, list) or len(turns) < 3:
            continue
        ok = []
        for t in turns:
            if not isinstance(t, dict):
                continue
            text = str(t.get("text", "")).strip()
            spk = str(t.get("spk", "A")).strip()[:1].upper()
            text = text.replace("\n", " ").strip()
            if not text or len(text) > 80:
                continue
            if any(ch in text for ch in "（）()【】[]<>#*`"):
                continue
            ok.append({"spk": spk if spk in ("A", "B") else "A", "text": text})
        if len(ok) >= 3:
            clean.append({"scene": str(d.get("scene", spec["scene"]))[:40], "turns": ok})
    if len(clean) < max(3, spec["n_dialogues"] // 2):
        raise ValueError(f"too few dialogues: {len(clean)}")
    return clean


def check_batch(dialogues, spec):
    turns = [t for d in dialogues for t in d["turns"]]
    issues = []
    if len(turns) < spec["n_dialogues"] * 4 * 0.8:
        issues.append(f"总句数只有 {len(turns)}，太少")
    texts = [t["text"] for t in turns]
    joined = "\n".join(texts)
    short = sum(1 for x in texts if len(x) <= 4)
    if short > max(3, len(texts) * 0.2):
        issues.append(f"2–4字短句有 {short} 句，超过 20%")
    if any(len(x) > 70 for x in texts):
        issues.append("有句子超过 70 字")
    if len(set(texts)) < len(texts) * 0.9:
        issues.append("批内重复句太多")

    if spec["kind"] == "numbers":
        pat = NUMBER_PATTERNS[spec["focus_number"]]
        hit = sum(1 for x in texts if re.search(pat, x))
        need = max(4, int(len(dialogues) * 0.7))
        if hit < need:
            issues.append(f"只有 {hit} 句出现{spec['focus_number']}，至少需要 {need} 句")

    if spec["kind"] == "poly":
        words = [w for w in spec["poly"][1].split("、") if w]
        distinct = [w for w in words if w in joined]
        total = sum(joined.count(w) for w in words)
        if len(distinct) < min(3, len(words)):
            issues.append(f"含“{spec['poly'][0]}”的词只用到 {('、'.join(distinct) or '无')}，需要至少 {min(3, len(words))} 个不同词")
        if total < 8:
            issues.append(f"目标词只出现 {total} 次，至少需要 8 次")

    if spec["kind"] == "mixed":
        unit = {"g", "gb", "mb", "tb", "kg", "km", "cm", "mm", "ml", "m", "l", "kmh", "mbps"}
        latin_turns = [x for x in texts if LATIN.search(x)]
        tokens = [t for x in texts for t in LATIN.findall(x) if len(t) >= 2 and t.lower() not in unit]
        pool_hits = [w for w in spec["english_pool"] if w.lower() in joined.lower()]
        if len(latin_turns) < 6 or len(tokens) < 8 or len(set(t.lower() for t in tokens)) < 5 or len(pool_hits) < 3:
            issues.append(
                f"英文覆盖不足：{len(latin_turns)} 句含英文、{len(set(t.lower() for t in tokens))} 个不同英文词，"
                f"指定词只出现 {('、'.join(pool_hits) or '无')}"
            )

    if spec["kind"] == "long":
        long_turns = sum(1 for x in texts if len(x) > 28)
        if long_turns < len(texts) * 0.35:
            issues.append(f"长句只有 {long_turns} 句，未达到 35%")

    return issues


def valid_batch(path):
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("ok") is True and data.get("dialogues")
    except Exception:
        return False


def run_one(spec, avoid_openings):
    path = RAW / f"{spec['id']}.json"
    if valid_batch(path):
        return spec["id"], "skip", 0
    base_prompt = build_prompt(spec, avoid_openings)
    prompt = base_prompt
    last_err = None
    for attempt in range(4):
        try:
            resp = call_api(prompt)
            content = resp["choices"][0]["message"]["content"]
            dialogues = parse_and_check(content, spec)
            issues = check_batch(dialogues, spec)
            if issues:
                last_err = "；".join(issues)
                prompt = base_prompt + "\n\n上一版没有满足以下要求：" + last_err + "。请重写全部对话，务必逐条满足，不要解释。"
                time.sleep(1)
                continue
            path.write_text(json.dumps({
                "ok": True, "spec": spec, "dialogues": dialogues,
                "usage": resp.get("usage"), "ts": time.time(),
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            turns = sum(len(d["turns"]) for d in dialogues)
            return spec["id"], "ok", turns
        except Exception as e:
            last_err = str(e)
            time.sleep(2 + attempt * 3)
    path.write_text(json.dumps({"ok": False, "spec": spec, "error": last_err}, ensure_ascii=False, indent=1), encoding="utf-8")
    return spec["id"], f"fail: {last_err}", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=160)
    ap.add_argument("--indices", type=str, default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shuffle-avoid", action="store_true")
    args = ap.parse_args()

    if args.indices:
        idxs = [int(x) for x in args.indices.split(",") if x.strip()]
    else:
        idxs = list(range(args.start, args.start + args.count))
    specs = [make_spec(i) for i in idxs]
    avoid_pool = ["我觉得", "那个", "您好", "你看", "其实", "就是说", "不是", "这个"]
    rng = random.Random(42)
    done = 0
    turns_total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for spec in specs:
            avoid = rng.sample(avoid_pool, 4)
            futs[ex.submit(run_one, spec, avoid)] = spec["id"]
        for fut in as_completed(futs):
            sid, status, turns = fut.result()
            done += 1
            turns_total += turns
            print(f"[{done}/{len(specs)}] {sid} {status} turns={turns} total={turns_total}", flush=True)
    print(f"DONE batches={done} turns={turns_total}", flush=True)


if __name__ == "__main__":
    main()
