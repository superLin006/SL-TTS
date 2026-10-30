import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

ROOT = Path("/mnt/data/itc_tts")
rows = [json.loads(x) for x in (ROOT / "pron_scan/student_input_pron_pinyin.jsonl").read_text(encoding="utf-8").splitlines()][:3]
mapping = json.loads((ROOT / "pron_scan/pron_mapping.json").read_text(encoding="utf-8"))[:3]
outdir = ROOT / "export/test_onnx"
outdir.mkdir(exist_ok=True)
for model in ["pinyin224.onnx", "pinyin224_int8.onnx"]:
    sess = ort.InferenceSession(str(ROOT / "export" / model), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    print(model, "inputs", names, flush=True)
    for i, (row, meta) in enumerate(zip(rows, mapping)):
        ids = row["phoneme_ids"]
        feed = {
            "input": np.array([ids], dtype=np.int64),
            "input_lengths": np.array([len(ids)], dtype=np.int64),
            "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32),
        }
        if "sid" in names:
            feed["sid"] = np.array([0], dtype=np.int64)
        t0 = time.perf_counter()
        audio = sess.run(None, feed)[0].squeeze()
        dt = time.perf_counter() - t0
        dur = len(audio) / 22050
        print(model, meta["word"], "chars", len(meta["text"]), "dur", round(dur, 2), "infer", round(dt, 3), "rtf", round(dt / dur, 3), "nan", not np.isfinite(audio).all(), flush=True)
        sf.write(str(outdir / f"{model}_{meta['word']}.wav"), audio, 22050)
