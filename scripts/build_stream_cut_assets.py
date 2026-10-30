import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
    quantize_dynamic, quantize_static,
)

ROOT = Path(__file__).resolve().parent.parent
CUT = "/Mul_7_output_0"

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(ROOT / "export/pinyin224.onnx"))
ap.add_argument("--out", default=str(ROOT / "export/stream_cut"))
ap.add_argument("--calib", default=str(ROOT / "data/calib_pinyin.jsonl"))
ap.add_argument("--smoke-input", default="")
args = ap.parse_args()
SRC = Path(args.src)
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

model = onnx.load(str(SRC))
assert any(CUT in n.output for n in model.graph.node), "latent cut missing"
latent = helper.make_tensor_value_info(CUT, TensorProto.FLOAT, [1, 192, "frames"])

reference = copy.deepcopy(model)
reference.graph.output.append(latent)
onnx.checker.check_model(reference)
onnx.save(reference, str(OUT / "reference.onnx"))

echo = copy.deepcopy(model)
echo.graph.CopyFrom(helper.make_graph([
    helper.make_node("Cast", ["input"], ["ids_float"], to=TensorProto.FLOAT),
    helper.make_node("Constant", [], ["axes"], value=helper.make_tensor("axis", TensorProto.INT64, [1], [1])),
    helper.make_node("Unsqueeze", ["ids_float", "axes"], ["output"]),
], "frontend_ids_only_do_not_play", list(model.graph.input),
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, "phonemes"])]))
del echo.metadata_props[:]
for k, v in {"sample_rate": "22050", "model_type": "vits", "language": "Chinese", "voice": "zh",
             "has_espeak": "0", "has_g2pw": "1", "n_speakers": "1", "comment": "piper", "version": "1"}.items():
    e = echo.metadata_props.add()
    e.key = k
    e.value = v
onnx.checker.check_model(echo)
onnx.save(echo, str(OUT / "frontend-ids-only.onnx"))


def extract(outputs, inputs, stop):
    result = copy.deepcopy(model)
    producers = {name: index for index, n in enumerate(model.graph.node) for name in n.output}
    needed, pending, selected = set(), [o.name for o in outputs], set()
    while pending:
        name = pending.pop()
        if name in needed or name in stop:
            continue
        needed.add(name)
        index = producers.get(name)
        if index is not None:
            selected.add(index)
            pending.extend(model.graph.node[index].input)
    nodes = [n for index, n in enumerate(model.graph.node) if index in selected]
    initializers = [i for i in model.graph.initializer if i.name in needed]
    result.graph.CopyFrom(helper.make_graph(nodes, "vits_stream_probe", inputs, outputs, initializers))
    onnx.checker.check_model(result)
    return result


encoder = extract([latent], list(model.graph.input), set())
onnx.save(encoder, str(OUT / "encoder_float.onnx"))
decoder = extract(list(model.graph.output), [latent], {CUT})
onnx.save(decoder, str(OUT / "decoder-float.onnx"))
print("sizes", {p.name: round(p.stat().st_size / 1e6, 1) for p in OUT.glob("*.onnx")}, flush=True)

feeds = []
seen = set()
for line in Path(args.calib).read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    ids = json.loads(line)["phoneme_ids"]
    key = tuple(ids)
    if key in seen:
        continue
    seen.add(key)
    arr = np.zeros((1, 320), dtype=np.int64)
    arr[0, :min(len(ids), 320)] = ids[:320]
    n = min(len(ids), 320)
    feeds.append({"input": arr, "input_lengths": np.array([n], dtype=np.int64),
                  "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32)})
print("calibration", len(feeds), flush=True)


class Reader(CalibrationDataReader):
    def __init__(self, data):
        self.data = iter(data)

    def get_next(self):
        return next(self.data, None)


def static(dst, op_types=None):
    quantize_static(str(OUT / "encoder_float.onnx"), str(dst), Reader(feeds),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax,
                    op_types_to_quantize=op_types)
    print("static", dst.name, round(dst.stat().st_size / 1e6, 1), "MB", flush=True)


static(OUT / "encoder_static_full.onnx")
static(OUT / "encoder.onnx", ["Conv"])
quantize_dynamic(str(OUT / "encoder_float.onnx"), str(OUT / "encoder_dynamic_u8.onnx"),
                 weight_type=QuantType.QUInt8, per_channel=True)
print("dynamic", round((OUT / "encoder_dynamic_u8.onnx").stat().st_size / 1e6, 1), "MB", flush=True)

if args.smoke_input:
    item = json.loads(Path(args.smoke_input).read_text(encoding="utf-8"))[0]
    ids = item.get("phoneme_ids") or item.get("ids")
    arr = np.zeros((1, 320), dtype=np.int64)
    arr[0, :min(len(ids), 320)] = ids[:320]
    feed = {"input": arr, "input_lengths": np.array([min(len(ids), 320)], dtype=np.int64),
            "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32)}
    dec = ort.InferenceSession(str(OUT / "decoder-float.onnx"), providers=["CPUExecutionProvider"])
    for name in ["encoder_float.onnx", "encoder_static_full.onnx", "encoder.onnx", "encoder_dynamic_u8.onnx"]:
        enc = ort.InferenceSession(str(OUT / name), providers=["CPUExecutionProvider"])
        lat = enc.run(None, feed)[0]
        audio = dec.run(None, {CUT: lat})[0].squeeze()
        print("smoke", name, lat.shape, round(len(audio) / 22050, 2),
              "finite", bool(np.isfinite(audio).all()), flush=True)
print("DONE", OUT, flush=True)
