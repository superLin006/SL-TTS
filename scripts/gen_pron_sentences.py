import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gen_dialogues as g

ROOT = Path(__file__).parent
ROOT.mkdir(exist_ok=True)
OUT = ROOT / "pron_sentences.jsonl"

WORDS = [
    "银行", "行走", "行长", "行业", "重量", "重复", "重新", "重庆",
    "还是", "归还", "还款", "还有", "音乐", "乐器", "快乐", "乐意",
    "长江", "长大", "长度", "增长", "觉得", "记得", "得到", "你得",
    "背包", "背诵", "背景", "后背", "空调", "调查", "调整", "调换",
    "出差", "参差", "差别", "差一点", "着急", "拿着", "穿着", "着火",
    "为了", "因为", "成为", "认为", "发现", "头发", "发票", "发型",
    "数字", "数一下", "数量", "有空", "空气", "空间", "干净", "干活",
    "干燥", "种子", "种花", "种类", "处理", "到处", "相处", "上传",
    "传记", "传统", "关系", "系鞋带", "角度", "角色", "效率", "率领",
    "方便", "便宜", "参加", "人参", "薄饼", "薄弱", "露天", "露面",
    "龃龉", "觊觎", "踟蹰", "蹒跚", "氤氲", "旖旎", "缱绻", "迤逦",
    "囹圄", "耄耋", "饕餮", "魑魅", "恣睢", "忸怩", "蹂躏", "罹难",
    "讪笑", "遴选", "蹙眉", "攥住",
]

PROMPT = """请为下面每个词写一句自然中文短句，句子必须原样包含该词，长度 10–30 个汉字，口语化，不要解释。

词语：{words}

只输出 JSON：
{{"sentences":[{{"word":"银行","text":"..."}}]}}
"""


def main():
    rows = []
    for start in range(0, len(WORDS), 35):
        chunk = WORDS[start:start + 35]
        prompt = PROMPT.format(words="、".join(chunk))
        for attempt in range(3):
            resp = g.call_api(prompt)
            data = json.loads(resp["choices"][0]["message"]["content"])
            got = {x.get("word"): str(x.get("text", "")).strip() for x in data.get("sentences", [])}
            missing = [w for w in chunk if w not in got or w not in got[w]]
            for w in chunk:
                t = got.get(w, "")
                if w in t and 8 <= len(t) <= 35:
                    rows.append({"word": w, "text": t})
            if not missing:
                break
            prompt = PROMPT.format(words="、".join(missing))
        print("chunk", start, "rows", len(rows), flush=True)

    # 去重并按词表顺序
    seen = set()
    out = []
    for w in WORDS:
        for r in rows:
            if r["word"] == w and w not in seen:
                seen.add(w)
                out.append({"id": f"R{len(out)+1:03d}", "word": w, "text": r["text"]})
                break
    with OUT.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("words", len(WORDS), "covered", len(out), "missing", [w for w in WORDS if w not in seen])


if __name__ == "__main__":
    main()
