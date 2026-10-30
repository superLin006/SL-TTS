import json
from pathlib import Path
from pypinyin import lazy_pinyin, Style
import frontend_zh


class PinyinFrontend:
    def __init__(self, tokens_path, lexicon_path):
        self.token_map = {}
        for line in Path(tokens_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sym, idx = line.rsplit(" ", 1)
            self.token_map[sym] = int(idx)
        self.lex = {}
        for line in Path(lexicon_path).read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word, toks = parts[0], parts[1:]
            if 1 <= len(word) <= 8 and all("\u4e00" <= c <= "\u9fff" for c in word):
                self.lex[word] = toks
        custom = Path(lexicon_path).with_name("custom_tokens.json")
        if custom.exists():
            self.lex.update(json.loads(custom.read_text(encoding="utf-8")))
        self.maxlen = max(len(w) for w in self.lex)

    def _syllable(self, ch):
        ini = lazy_pinyin(ch, style=Style.INITIALS, strict=False,
                          neutral_tone_with_five=True)[0]
        fin = lazy_pinyin(ch, style=Style.FINALS_TONE3, strict=False,
                          neutral_tone_with_five=True)[0]
        tone = fin[-1] if fin and fin[-1] in "12345" else "5"
        fin = fin[:-1] if fin and fin[-1] in "12345" else fin
        toks = []
        toks.append(ini if ini else "Ø")
        if fin:
            toks.append(fin)
        toks.append(tone)
        toks.append("_")
        return toks

    def tokens(self, raw):
        text, _ = frontend_zh.normalize_text(raw)
        out = ["^"]
        i = 0
        while i < len(text):
            ch = text[i]
            if "\u4e00" <= ch <= "\u9fff":
                hit = None
                for n in range(min(self.maxlen, len(text) - i), 0, -1):
                    w = text[i:i + n]
                    if w in self.lex:
                        hit = (w, self.lex[w])
                        break
                if hit:
                    out.extend(hit[1])
                    i += len(hit[0])
                else:
                    out.extend(self._syllable(ch))
                    i += 1
            else:
                if ch in self.token_map:
                    out.append(ch)
                i += 1
        out.append("$")
        compact = []
        for t in out:
            if t == "_" and compact and compact[-1] == "_":
                continue
            compact.append(t)
        return compact

    def ids(self, raw):
        toks = self.tokens(raw)
        missing = [t for t in toks if t not in self.token_map]
        return [self.token_map[t] for t in toks if t in self.token_map], missing, toks
