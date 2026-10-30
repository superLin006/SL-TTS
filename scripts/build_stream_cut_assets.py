import copy
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
)

ROOT = Path("/mnt/d/lvb-tools/tts-export")
SRC = ROOT / "pinyin224.onnx"
OUT = ROOT / "stream_cut"
OUT.mkdir(exist_ok=True)
cut = "/Mul_7_output_0"

model = onnx.load(str(SRC))
assert any(cut in n.output for n in model.graph.node), "latent cut missing"
latent = helper.make_tensor_value_info(cut, TensorProto.FLOAT, [1, 192, "frames"])

# reference with latent output (diagnostic)
reference = copy.deepcopy(model)
reference.graph.output.append(latent)
onnx.checker.check_model(reference)
onnx.save(reference, str(OUT / "reference.onnx"))

# frontend echo model: real sherpa frontend, returns token IDs as floats
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
    e = echo.metadata_props.add(); e.key = k; e.value = v
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
decoder = extract(list(model.graph.output), [latent], {cut})
onnx.save(decoder, str(OUT / "decoder-float.onnx"))
print("encoder inputs", [i.name for i in encoder.graph.input], "outputs", [o.name for o in encoder.graph.output])
print("decoder inputs", [i.name for i in decoder.graph.input], "outputs", [o.name for o in decoder.graph.output])
print("sizes", {p.name: round(p.stat().st_size/1e6,1) for p in OUT.glob("*.onnx")})

# quantize encoder static QDQ with padded calibration
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

quantize_static(str(OUT / "encoder_float.onnx"), str(OUT / "encoder.onnx"), Reader(feeds),
                quant_format=QuantFormat.QDQ, per_channel=True,
                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                calibrate_method=CalibrationMethod.MinMax)
print("encoder_int8", round((OUT/'encoder.onnx').stat().st_size/1e6,1), "MB")

# end-to-end smoke
enc = ort.InferenceSession(str(OUT / "encoder.onnx"), providers=["CPUExecutionProvider"])
dec = ort.InferenceSession(str(OUT / "decoder-float.onnx"), providers=["CPUExecutionProvider"])
item = json.loads((ROOT / "inputs.json").read_text(encoding="utf-8"))[0]
ids = item["ids"]
arr = np.zeros((1, 320), dtype=np.int64); arr[0, :len(ids)] = ids
feed = {"input": arr, "input_lengths": np.array([len(ids)], dtype=np.int64),
        "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32)}
lat = enc.run(None, feed)[0]
audio = dec.run(None, {cut: lat})[0].squeeze()
print("latent", lat.shape, "audio", audio.shape, "dur", round(len(audio)/22050,2), "finite", np.isfinite(audio).all())
import wave
pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
with wave.open(str(OUT / "smoke.wav"), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050); w.writeframes(pcm.tobytes())
