import re

DIGITS = "零一二三四五六七八九"
BIG_UNITS = ["", "万", "亿", "万亿"]

EN_TERMS = [
    ("my.email@example.com", "电子邮件地址"),
    ("email@example.com", "电子邮件地址"),
    ("PowerPoint", "幻灯片"),
    ("WiFi", "无线网络"),
    ("CSV", "表格文件"),
    ("Micro-USB", "微型优艾斯比接口"),
    ("USB-C", "优艾斯比西接口"),
    ("USB", "优艾斯比接口"),
    ("HDMI", "高清多媒体接口"),
    ("Windows", "视窗系统"),
    ("Excel", "表格"),
    ("Office", "办公软件"),
    ("DNS", "域名解析"),
    ("PIN码", "密码"),
    ("PIN", "个人识别码"),
    ("GPU", "图形处理器"),
    ("App", "应用"),
    ("APP", "应用"),
    ("app", "应用"),
    ("ZIP", "压缩包"),
    ("PVC", "聚氯乙烯"),
    ("SN", "序列号"),
    ("AA制", "各付各的"),
    ("AA", "各付各的"),
    ("iPhone", "苹果手机"),
    ("Wi-Fi", "无线网络"),
    ("Bluetooth", "蓝牙"),
    ("Android", "安卓"),
    ("iOS", "苹果系统"),
    ("vivo", "维沃"),
    ("OPPO", "欧珀"),
    ("PDF", "批迪艾弗文件"),
    ("PPT", "幻灯片"),
    ("API", "接口"),
    ("CPU", "西批优"),
    ("VPN", "虚拟专用网络"),
    ("5G", "第五代移动通信"),
    ("AI", "人工智能"),
    ("email", "电子邮件"),
    ("Alice", "爱丽丝"),
    ("timeout", "超时"),
    ("retry", "重试"),
    ("client", "客户"),
    ("privacy", "隐私"),
    ("invoice", "发票"),
    ("contract", "合同"),
    ("adapter", "转接头"),
    ("bug", "缺陷"),
    ("Pro", "专业版"),
    ("A股", "诶股"),
]

LETTERS = {
    "A": "诶", "B": "比", "C": "西", "D": "迪", "E": "伊", "F": "艾弗",
    "G": "吉", "H": "艾尺", "I": "艾", "J": "杰", "K": "开", "L": "艾勒",
    "M": "艾姆", "N": "艾恩", "O": "欧", "P": "批", "Q": "丘", "R": "啊",
    "S": "艾斯", "T": "提", "U": "优", "V": "维", "W": "达不溜",
    "X": "艾克斯", "Y": "歪", "Z": "贼德",
}


def _section(n):
    out = ""
    zero = False
    for unit, div in (("千", 1000), ("百", 100), ("十", 10), ("", 1)):
        d = (n // div) % 10
        if d == 0:
            if out and not zero:
                zero = True
        else:
            if zero:
                out += "零"
                zero = False
            out += DIGITS[d] + unit
    return out


def int_to_zh(n):
    n = int(n)
    if n == 0:
        return "零"
    sections = []
    while n > 0:
        sections.append(n % 10000)
        n //= 10000
    parts = []
    zero_pending = False
    for idx in range(len(sections) - 1, -1, -1):
        sec = sections[idx]
        if sec == 0:
            if parts:
                zero_pending = True
            continue
        part = _section(sec)
        if parts and (zero_pending or sec < 1000):
            part = "零" + part
        parts.append(part + BIG_UNITS[idx])
        zero_pending = False
    s = "".join(parts)
    if s.startswith("一十"):
        s = s[1:]
    return s


def number_to_zh(s):
    s = s.replace(",", "").replace("，", "").strip()
    neg = s.startswith("-")
    s = s.lstrip("+-")
    if "." in s:
        a, b = s.split(".", 1)
        out = (int_to_zh(int(a)) if a else "零") + "点" + "".join(DIGITS[int(c)] for c in b if c.isdigit())
    else:
        out = int_to_zh(int(s))
    return ("负" + out) if neg else out


def digits_to_zh(s):
    return "".join(DIGITS[int(c)] for c in s if c.isdigit())


def year_to_zh(s):
    return "".join(DIGITS[int(c)] for c in s if c.isdigit())


def _latin_word(m):
    return "".join(LETTERS.get(c.upper(), c) for c in m.group(0))


TECH_UNITS = [
    ("Gbps", "吉比特每秒"), ("Mbps", "兆比特每秒"), ("Kbps", "千比特每秒"),
    ("km/h", "公里每小时"), ("m/s", "米每秒"), ("°C", "摄氏度"), ("℃", "摄氏度"),
    ("GHz", "吉赫兹"), ("MHz", "兆赫兹"), ("Hz", "赫兹"),
    ("mAh", "毫安时"), ("Wh", "瓦时"), ("kW", "千瓦"),
    ("GB", "吉字节"), ("TB", "太字节"), ("MB", "兆字节"), ("KB", "千字节"),
    ("ms", "毫秒"), ("W", "瓦"), ("V", "伏"), ("A", "安"),
    ("G", "吉字节"), ("T", "太字节"), ("M", "兆字节"), ("K", "千字节"),
]


def _normalize_tech_units(text, notes):
    for unit, zh in sorted(TECH_UNITS, key=lambda x: -len(x[0])):
        def repl(m):
            num, duo = m.group(1), m.group(2) or ""
            out = number_to_zh(num) + duo + zh
            notes.append(f"{m.group(0)}→{out}")
            return out

        text = re.sub(
            r"(\d+(?:\.\d+)?)\s*(多)?\s*" + re.escape(unit) + r"(?![A-Za-z])" + (r"(?!级)" if unit == "A" else ""),
            repl, text,
        )
    return text


def _replace_english(text, notes):
    for old, new in EN_TERMS:
        if old in text:
            text = text.replace(old, new)
            notes.append(f"{old}→{new}")

    text = _normalize_tech_units(text, notes)

    def flight(m):
        letters, num = m.group(1), m.group(2)
        out = "".join(LETTERS.get(c, c) for c in letters) + digits_to_zh(num)
        notes.append(f"{letters}{num}→{out}")
        return out

    text = re.sub(r"(?<![A-Za-z0-9])([A-Z]{2})(\d{3,4})(?![A-Za-z0-9])", flight, text)

    def train(m):
        prefix, num = m.group(1), m.group(2)
        out = {"G": "高", "D": "动", "C": "城", "Z": "直", "T": "特", "K": "快"}[prefix] + digits_to_zh(num)
        notes.append(f"{m.group(0)}→{out}")
        return out

    text = re.sub(r"(?<![A-Za-z0-9])([GDCZTK])(\d{1,4})(?![A-Za-z0-9])", train, text)

    def email(m):
        notes.append("email→电子邮件地址")
        return "电子邮件地址"

    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", email, text)

    def latin(m):
        notes.append(f"{m.group(0)}→按字母读")
        return _latin_word(m)

    text = re.sub(r"(?<![A-Za-z0-9])([A-Za-z]{2,})(?![A-Za-z0-9])", latin, text)
    text = re.sub(r"(?<=\d)([A-Za-z]{1,3})(?![A-Za-z0-9])", latin, text)
    text = re.sub(r"(?<![A-Za-z0-9])([A-Za-z]{1,2})(?=\d)", latin, text)
    text = re.sub(r"(?<![A-Za-z0-9])([A-Za-z])(?![A-Za-z0-9])", latin, text)
    text = text.replace("+", "加")
    return text


def _normalize_dates(text, notes):
    def y(m):
        out = year_to_zh(m.group(1)) + "年"
        notes.append(f"{m.group(0)}→{out}")
        return out

    text = re.sub(r"(\d{4})\s*年", y, text)

    def md(m):
        num, unit = m.group(1), m.group(2)
        out = int_to_zh(int(num)) + unit
        notes.append(f"{m.group(0)}→{out}")
        return out

    text = re.sub(r"(\d{1,2})\s*(月|日|号)", md, text)
    return text


def _normalize_times(text, notes):
    def hm(m):
        h, mi = int(m.group(1)), int(m.group(2))
        out = int_to_zh(h) + "点" + (int_to_zh(mi) + "分" if mi else "整")
        notes.append(f"{m.group(0)}→{out}")
        return out

    text = re.sub(r"(\d{1,2}):(\d{2})", hm, text)

    def hm_cn(m):
        h, mi = int(m.group(1)), int(m.group(2))
        out = int_to_zh(h) + "点" + int_to_zh(mi) + "分"
        notes.append(f"{m.group(0)}→{out}")
        return out

    text = re.sub(r"(\d{1,2})\s*点\s*(\d{1,2})\s*分", hm_cn, text)

    def half(m):
        h = int(m.group(1))
        out = int_to_zh(h) + "点半"
        notes.append(f"{m.group(0)}→{out}")
        return out

    text = re.sub(r"(\d{1,2})\s*点\s*半", half, text)

    def hour(m):
        h = int(m.group(1))
        out = int_to_zh(h) + "点"
        notes.append(f"{m.group(0)}→{out}")
        return out

    text = re.sub(r"(\d{1,2})\s*点", hour, text)
    return text


def _normalize_special(text, notes):
    def ip(m):
        out = "点".join(digits_to_zh(g) for g in m.groups())
        notes.append(f"{m.group(0)}→{out}")
        return out

    text = re.sub(r"(?<![\d.])(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?![\d.])", ip, text)

    def dial(m):
        digits = re.sub(r"\D", "", m.group(1))
        out = "拨打" + digits_to_zh(digits)
        notes.append(f"{m.group(0)}→{out}")
        return out

    text = re.sub(r"拨打\s*([\d\s]{3,})", dial, text)

    def phone(m):
        digits = re.sub(r"\D", "", m.group(0))
        out = digits_to_zh(digits)
        notes.append(f"{m.group(0)}→{out}")
        return out

    text = re.sub(r"(?<!\d)\d{3}\s+\d{4}\s+\d{4}(?!\d)", phone, text)
    text = re.sub(r"(?<!\d)\d{11}(?!\d)", phone, text)
    text = re.sub(r"(?<!\d)\d{7,}(?!\d)", phone, text)

    def after(m):
        prefix, num = m.group(1), re.sub(r"\D", "", m.group(2))
        out = prefix + digits_to_zh(num)
        notes.append(f"{m.group(0)}→{out}")
        return out

    text = re.sub(r"(后四位|编号|订单编号)\s*(?:是|为|号)?\s*(\d{2,})", after, text)

    def room(m):
        out = digits_to_zh(m.group(1)) + "室"
        notes.append(f"{m.group(0)}→{out}")
        return out

    text = re.sub(r"(?<!\d)(\d{3,4})\s*室", room, text)
    return text


def _normalize_measures(text, notes):
    units = "亿元|万元|元|块钱|块|公里|千米|米|千克|公斤|件|个|年|天|小时|分钟|分|秒|岁|人|次|台|套|倍|厘米|毫米|度|折"
    num = r"[+-]?\d[\d,]*(?:\.\d+)?"

    def pct(m):
        out = "百分之" + number_to_zh(m.group(1))
        notes.append(f"{m.group(0)}→{out}")
        return out

    text = re.sub(r"(" + num + r")\s*%", pct, text)

    def unit(m):
        raw, u = m.group(1), m.group(2)
        out = number_to_zh(raw) + u
        notes.append(f"{m.group(0)}→{out}")
        return out

    text = re.sub(r"(" + num + r")\s*(" + units + r")", unit, text)

    def op(m):
        out = number_to_zh(m.group(1)) + m.group(2)
        notes.append(f"{m.group(0)}→{out}")
        return out

    text = re.sub(r"(" + num + r")\s*(乘|减|加|除以|等于)", op, text)
    return text


def _normalize_generic(text, notes):
    def num(m):
        out = number_to_zh(m.group(0))
        notes.append(f"{m.group(0)}→{out}")
        return out

    text = re.sub(r"[+-]?\d[\d,]*(?:\.\d+)?", num, text)
    return text


def normalize_text(raw):
    notes = []
    t = raw
    t = _replace_english(t, notes)
    t = _normalize_dates(t, notes)
    t = _normalize_times(t, notes)
    t = _normalize_special(t, notes)
    t = _normalize_measures(t, notes)
    t = _normalize_generic(t, notes)
    t = re.sub(r"\s+", "", t)
    t = re.sub(r"\s*([，。！？、；：])", r"\1", t)
    return t, notes


EN_SPAN = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"[A-Za-z][A-Za-z0-9.@+_%:/#-]*(?:\s+[A-Za-z0-9][A-Za-z0-9.@+_%:/#-]*)*"
    r"|\d+[A-Za-z][A-Za-z0-9.@+_%:/#-]*"
    r")"
)


def normalize_text_mixed(raw):
    """Keep English words for the English G2P; normalize only the Chinese/number parts."""
    notes = []
    t = raw

    def email(m):
        notes.append("email→my email at example dot com")
        return "my email at example dot com"

    t = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", email, t)

    spans = []
    def protect(m):
        spans.append(m.group(0))
        return "\uE000" + chr(0xE100 + len(spans) - 1) + "\uE001"

    t = EN_SPAN.sub(protect, t)
    t = _normalize_dates(t, notes)
    t = _normalize_times(t, notes)
    t = _normalize_special(t, notes)
    t = _normalize_measures(t, notes)
    t = _normalize_generic(t, notes)
    t = re.sub(r"\s+", "", t)
    t = re.sub(r"\s*([，。！？、；：])", r"\1", t)
    for i, span in enumerate(spans):
        t = t.replace("\uE000" + chr(0xE100 + i) + "\uE001", " " + span + " ")
    t = re.sub(r"\s+([，。！？、；：])", r"\1", t)
    t = re.sub(r"([，。！？、；：])\s+", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t, notes


def install_polyphone_dict():
    from pypinyin import load_phrases_dict

    phrases = {
        "银行": [["yin2"], ["hang2"]],
        "行长": [["hang2"], ["zhang3"]],
        "行规": [["hang2"], ["gui1"]],
        "行行": [["hang2"], ["hang2"]],
        "行业": [["hang2"], ["ye4"]],
        "长江": [["chang2"], ["jiang1"]],
        "长大": [["zhang3"], ["da4"]],
        "很长": [["hen3"], ["chang2"]],
        "重庆": [["chong2"], ["qing4"]],
        "重要": [["zhong4"], ["yao4"]],
        "重新": [["chong2"], ["xin1"]],
        "重口味": [["zhong4"], ["kou3"], ["wei4"]],
        "音乐": [["yin1"], ["yue4"]],
        "乐器": [["yue4"], ["qi4"]],
        "乐要": [["yue4"], ["yao4"]],
        "快乐": [["kuai4"], ["le4"]],
        "还了书": [["huan2"], ["le5"], ["shu1"]],
        "还了钱": [["huan2"], ["le5"], ["qian2"]],
        "还是": [["hai2"], ["shi4"]],
        "还有": [["hai2"], ["you3"]],
        "还很大": [["hai2"], ["hen3"], ["da4"]],
        "你得": [["ni3"], ["dei3"]],
        "写得": [["xie3"], ["de5"]],
        "演得": [["yan3"], ["de5"]],
        "背包": [["bei1"], ["bao1"]],
        "背在": [["bei1"], ["zai4"]],
        "调整": [["tiao2"], ["zheng3"]],
        "自行车": [["zi4"], ["xing2"], ["che1"]],
        "演员": [["yan3"], ["yuan2"]],
        "质量": [["zhi4"], ["liang4"]],
    }
    load_phrases_dict(phrases)
    return phrases


if __name__ == "__main__":
    import json
    from pathlib import Path

    rows = [json.loads(x) for x in Path("pilot_corpus_200.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    keys = {"p111", "p113", "p119", "p121", "p123", "p129", "p135", "p139", "p161", "p162", "p163", "p164", "p165", "p178", "p191", "p192", "p193", "p194", "p199", "p200"}
    for r in rows:
        if r["id"] in keys:
            print("---", r["id"], r["text"])
            print(normalize_text(r["text"])[0])
    for n in [0, 10, 12, 100, 101, 110, 1300, 3600, 12345, 123456, 1299, 10001, 10010]:
        print(n, int_to_zh(n))
