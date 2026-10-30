import json
import struct
import time
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path("/mnt/d/lvb-tools/tts-export")
items = json.loads((ROOT / "inputs.json").read_text(encoding="utf-8"))
for model in ["pinyin224.onnx", "pinyin224_static_int8.onnx", "pinyin224_dynamic_int8.onnx"]:
    sess = ort.InferenceSession(str(ROOT / model), providers=["CPUExecutionProvider"])
    for i, item in enumerate(items):
        ids = item["ids"]
        arr = np.zeros((1, 256), dtype=np.int64)
        arr[0, :len(ids)] = ids
        feed = {"input": arr, "input_lengths": np.array([len(ids)], dtype=np.int64),
                "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32)}
        t0 = time.perf_counter()
        audio = sess.run(None, feed)[0].squeeze()
        dt = time.perf_counter() - t0
        pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        out = ROOT / "android_test" / f"{model.split('.')[0]}_{i}.wav"
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050); w.writeframes(pcm.tobytes())
        print(model, i, "ids", len(ids), "dur", round(len(audio)/22050, 2), "rtf", round(dt/(len(audio)/22050), 3), flush=True)
