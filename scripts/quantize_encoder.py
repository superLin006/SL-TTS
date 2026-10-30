import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
)

ROOT = Path("/mnt/d/lvb-tools/tts-export")
SRC = ROOT / "streaming/encoder.onnx"
DST = ROOT / "streaming/encoder_int8.onnx"
feeds = []
seen = set()
for name in ["student_input_pron_pinyin.jsonl", "student_input_long_pinyin.jsonl", "student_input_200_pinyin.jsonl"]:
    for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ids = json.loads(line)["phoneme_ids"]
        key = tuple(ids)
        if key in seen:
            continue
        seen.add(key)
        arr = np.zeros((1, 320), dtype=np.int64)
        arr[0, :len(ids)] = ids
        feeds.append({"input": arr, "input_lengths": np.array([len(ids)], dtype=np.int64),
                      "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32)})
print("calibration", len(feeds), flush=True)

class Reader(CalibrationDataReader):
    def __init__(self, data):
        self.data = iter(data)
    def get_next(self):
        return next(self.data, None)

quantize_static(str(SRC), str(DST), Reader(feeds),
                quant_format=QuantFormat.QDQ, per_channel=True,
                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                calibrate_method=CalibrationMethod.MinMax)
print("size", round(DST.stat().st_size / 1e6, 1), "MB", flush=True)
m = onnx.load(str(DST))
ops = {}
for n in m.graph.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
print("ops", {k: ops[k] for k in ["QuantizeLinear", "DequantizeLinear", "ConvInteger"] if k in ops}, flush=True)
sess = ort.InferenceSession(str(DST), providers=["CPUExecutionProvider"])
item = json.loads((ROOT / "inputs.json").read_text(encoding="utf-8"))[0]
ids = item["ids"]
arr = np.zeros((1, 320), dtype=np.int64); arr[0, :len(ids)] = ids
feed = {"input": arr, "input_lengths": np.array([len(ids)], dtype=np.int64),
        "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32)}
outs = sess.run(None, feed)
for i, o in enumerate(outs):
    print("out", i, o.shape, "finite", np.isfinite(o).all(), flush=True)
