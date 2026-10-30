import argparse
from pypinyin import load_phrases_dict
import frontend_zh
from kokoro import KModel, KPipeline
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--add", action="store_true")
args = ap.parse_args()
frontend_zh.install_polyphone_dict()
MODEL = Path("/mnt/data/itc_tts/models/kokoro-v1.1-zh")
model = KModel(repo_id="hexgrad/Kokoro-82M-v1.1-zh", config=str(MODEL / "config.json"),
               model=str(MODEL / "kokoro-v1_1-zh.pth")).to("cuda").eval()
import torch
voice = torch.load(MODEL / "voices/zf_001.pt", map_location="cpu", weights_only=True)
pipe = KPipeline(lang_code="z", repo_id="hexgrad/Kokoro-82M-v1.1-zh", model=model)
if args.add:
    import jieba
    jieba.add_word("还书", freq=100000)
    jieba.add_word("数一下", freq=100000)
    jieba.add_word("上传", freq=100000)
    load_phrases_dict({
        "还书": [["huan2"], ["shu1"]],
        "系鞋带": [["ji4"], ["xie2"], ["dai4"]],
        "种花": [["zhong4"], ["hua1"]],
    })
for text in ["我明天去图书馆还书。", "你鞋带松了，赶紧蹲下系鞋带吧。", "我妈特别喜欢在阳台上种花。"]:
    for gs, ps, audio in pipe(text, voice=voice):
        print("PHON", ps[:120], flush=True)
