import json
import logging
import os
import sys

import piper_train.preprocess as pp
from pinyin_frontend import PinyinFrontend

TOKENS = "/mnt/data/itc_tts/pinyin/tokens.txt"
LEXICON = "/mnt/data/itc_tts/pinyin/lexicon.txt"


def worker(args, queue_in, queue_out):
    logger = logging.getLogger("preprocess_pinyin")
    pf = PinyinFrontend(TOKENS, LEXICON)
    silence_detector = pp.make_silence_detector()
    while True:
        batch = queue_in.get()
        if batch is None:
            break
        for utt in batch:
            try:
                ids, missing, toks = pf.ids(utt.text)
                utt.phonemes = toks
                utt.phoneme_ids = ids
                for m in missing:
                    utt.missing_phonemes[m] += 1
                if not args.skip_audio:
                    utt.audio_norm_path, utt.audio_spec_path = pp.cache_norm_audio(
                        utt.audio_path, args.cache_dir, silence_detector, args.sample_rate
                    )
                queue_out.put(utt)
            except Exception:
                logger.exception("Failed to process utterance: %s", utt)
                queue_out.put(None)
        queue_in.task_done()


pp.phonemize_batch_espeak = worker
pp.main()

# 用 App 的 pinyin token 表覆盖 preprocess 写的 espeak 配置
cfg_path = os.path.join(sys.argv[sys.argv.index("--output-dir") + 1], "config.json")
with open(cfg_path, encoding="utf-8") as f:
    cfg = json.load(f)
token_map = {}
for line in open(TOKENS, encoding="utf-8"):
    if line.strip():
        sym, idx = line.rsplit(" ", 1)
        token_map[sym] = int(idx)
cfg["phoneme_id_map"] = token_map
cfg["num_symbols"] = max(token_map.values()) + 1
cfg["phoneme_type"] = "text"
cfg["espeak"] = {"voice": "pinyin"}
cfg["language"] = {"code": "zh"}
with open(cfg_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=4)
print("config updated", cfg["num_symbols"], "symbols", flush=True)
