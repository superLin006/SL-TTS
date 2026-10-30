import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
)

ROOT = Path("/mnt/d/lvb-tools/tts-export")
src = ROOT / "pinyin224.onnx"
dst = ROOT / "pinyin224_static_int8.onnx"
rows = [json.loads(x) for x in (ROOT / "student_input_pron_pinyin.jsonl").read_text(encoding="utf-8").splitlines()][:80]
feeds = []
for r in rows:
    ids = r["phoneme_ids"]
    arr = np.zeros((1, 256), dtype=np.int64)
    arr[0, :len(ids)] = ids
    feeds.append({"input": arr,
                  "input_lengths": np.array([len(ids)], dtype=np.int64),
                  "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32)})

class Reader(CalibrationDataReader):
    def __init__(self, data):
        self.data = iter(data)
    def get_next(self):
        return next(self.data, None)

quantize_static(str(src), str(dst), Reader(feeds),
                quant_format=QuantFormat.QDQ, per_channel=True,
                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                calibrate_method=CalibrationMethod.MinMax)
print("size", round(dst.stat().st_size / 1e6, 1), "MB", flush=True)
m = onnx.load(str(dst))
ops = {}
for n in m.graph.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
print("ops", {k: ops[k] for k in ["QLinearConv", "ConvInteger", "MatMulInteger", "QuantizeLinear", "DequantizeLinear"] if k in ops}, flush=True)
sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
for i, r in enumerate(rows[:3]):
    ids = r["phoneme_ids"]
    arr = np.zeros((1, 256), dtype=np.int64); arr[0, :len(ids)] = ids
    feed = {"input": arr, "input_lengths": np.array([len(ids)], dtype=np.int64),
            "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32)}
    t0 = time.perf_counter(); audio = sess.run(None, feed)[0].squeeze(); dt = time.perf_counter() - t0
    print(i, "len", len(ids), "dur", round(len(audio)/22050, 2), "rtf", round(dt/(len(audio)/22050), 3), "nan", not np.isfinite(audio).all(), flush=True)
