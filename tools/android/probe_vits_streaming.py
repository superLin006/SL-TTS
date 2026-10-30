"""Offline feasibility probe, NOT a streaming TTS backend or a speech-quality test.

export: use an environment with onnx; read the existing model archive without changing it.
probe: use an environment with numpy/onnxruntime. Synthetic phoneme IDs exercise the graph;
reported desktop timings must never be presented as Android performance.
"""
import argparse
import json
import pathlib
import time


def export(args):
    import copy
    import tarfile
    import onnx
    from onnx import helper, TensorProto

    with tarfile.open(args.archive) as archive:
        model = onnx.load_model_from_string(archive.extractfile(args.member).read())
    # Deliberately tied to the inspected Piper export; fail rather than guess another cut.
    cut = "/Mul_7_output_0"
    assert any(cut in n.output for n in model.graph.node), "Unsupported graph: latent cut missing"
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), "Use a new empty output directory"
    latent = helper.make_tensor_value_info(cut, TensorProto.FLOAT, [1, 192, "frames"])
    reference = copy.deepcopy(model)
    reference.graph.output.append(latent)
    onnx.checker.check_model(reference)
    onnx.save(reference, out / "reference.onnx")
    # Diagnostic model: use the real sherpa frontend, but return its exact input
    # IDs as floats instead of speech. NEVER play this output as audio.
    echo = copy.deepcopy(model)
    echo.graph.CopyFrom(helper.make_graph([
        helper.make_node("Cast", ["input"], ["ids_float"], to=TensorProto.FLOAT),
        helper.make_node("Constant", [], ["axes"],
                         value=helper.make_tensor("axis", TensorProto.INT64, [1], [1])),
        helper.make_node("Unsqueeze", ["ids_float", "axes"], ["output"]),
    ], "frontend_ids_only_do_not_play", list(model.graph.input),
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, "phonemes"])]))
    onnx.checker.check_model(echo)
    onnx.save(echo, out / "frontend-ids-only.onnx")
    with tarfile.open(args.archive) as archive:
        prefix = str(pathlib.PurePosixPath(args.member).parent)
        for name in ("tokens.txt", "lexicon.txt", "phone.fst", "date.fst", "number.fst"):
            if prefix + "/" + name in archive.getnames():
                (out / name).write_bytes(archive.extractfile(prefix + "/" + name).read())
        for member in archive.getmembers():
            if not member.isfile() or not member.name.startswith(prefix + "/espeak-ng-data/"):
                continue
            relative = pathlib.PurePosixPath(member.name[len(prefix)+1:])
            assert not relative.is_absolute() and ".." not in relative.parts
            destination = out.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.extractfile(member).read())

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

    onnx.save(extract([latent], list(model.graph.input), set()), out / "encoder.onnx")
    decoder = extract(list(model.graph.output), [latent], {cut})
    onnx.save(decoder, out / "decoder.onnx")
    print(json.dumps({"nodes": len(model.graph.node), "decoder_nodes": len(decoder.graph.node),
                      "dynamic_quantizers": sum(n.op_type == "DynamicQuantizeLinear" for n in decoder.graph.node),
                      "cut": cut, "output": str(out)}, indent=2))


def capture(args):
    import sherpa_onnx
    import numpy as np
    folder = pathlib.Path(args.output)
    assert args.text and args.tokens
    target = pathlib.Path(args.tokens)
    assert not target.exists(), "Refuse to overwrite token evidence"
    plain_target = target.with_suffix(".ids.txt")
    assert not plain_target.exists(), "Refuse to overwrite token evidence"
    engine = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(folder / "frontend-ids-only.onnx"),
            lexicon=str(folder / "lexicon.txt") if (folder / "lexicon.txt").exists() else "",
            data_dir=str(folder / "espeak-ng-data") if (folder / "espeak-ng-data").is_dir() else "",
            tokens=str(folder / "tokens.txt")), num_threads=2),
        rule_fsts=",".join(str(folder / name) for name in ("phone.fst", "date.fst", "number.fst") if (folder / name).exists()),
        max_num_sentences=-1))
    samples = np.asarray(engine.generate(args.text, sid=0, speed=1).samples)
    assert samples.size and np.isfinite(samples).all() and np.equal(samples, np.round(samples)).all()
    ids = samples.astype(np.int64).tolist()
    valid_ids = {int(line.rsplit(maxsplit=1)[-1]) for line in
                 (folder / "tokens.txt").read_text(encoding="utf-8").splitlines() if line.strip()}
    assert set(ids) <= valid_ids, "Frontend echo returned invalid IDs"
    target.write_text(json.dumps({"text": args.text, "frontend_version": sherpa_onnx.__version__,
                                 "ids": ids}, ensure_ascii=False), encoding="utf-8")
    plain_target.write_text(" ".join(map(str, ids)), encoding="ascii")
    print(json.dumps({"frontend_version": sherpa_onnx.__version__, "token_count": len(ids),
                      "sample_rate": engine.sample_rate, "tokens": str(target)}))


def float_decoder(args):
    """Diagnostic only: dequantize stored weights, remove per-window activation quantization.

    This does NOT recover the original pre-quantization model or guarantee quality.
    Compare chunked/full output of this same variant to isolate boundary effects.
    """
    import onnx
    import numpy as np
    from onnx import helper, numpy_helper
    folder = pathlib.Path(args.output)
    target = folder / "decoder-float.onnx"
    assert not target.exists()
    model = onnx.load(folder / "decoder.onnx")
    nodes = list(model.graph.node)
    producers = {out: n for n in nodes for out in n.output}
    consumers = {}
    for node in nodes:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    initializers = {i.name: i for i in model.graph.initializer}
    replacements = {}
    for node in nodes:
        if node.op_type != "ConvInteger":
            continue
        quant = producers[node.input[0]]
        assert quant.op_type == "DynamicQuantizeLinear"
        cast, = consumers[node.output[0]]
        assert cast.op_type == "Cast"
        scaled, = consumers[cast.output[0]]
        assert scaled.op_type == "Mul"
        scale_node = producers[next(x for x in scaled.input if x != cast.output[0])]
        assert scale_node.op_type == "Mul" and quant.output[1] in scale_node.input
        scale_name = next(x for x in scale_node.input if x != quant.output[1])
        scale = numpy_helper.to_array(initializers[scale_name])
        zero = numpy_helper.to_array(initializers[node.input[3]])
        assert scale.size == zero.size == 1, "Only inspected per-tensor weight quantization supported"
        weight = (numpy_helper.to_array(initializers[node.input[1]]).astype(np.float32)-zero)*scale
        weight_name = node.input[1] + "_dequantized"
        initializers[weight_name] = numpy_helper.from_array(weight, weight_name)
        replacement = helper.make_node("Conv", [quant.input[0], weight_name], list(scaled.output),
                                       name=node.name + "_float_probe")
        replacement.attribute.extend(node.attribute)
        replacements[scaled.output[0]] = replacement
    nodes = [replacements.get(n.output[0], n) for n in nodes]
    producers = {out: index for index, n in enumerate(nodes) for out in n.output}
    pending = [o.name for o in model.graph.output]
    needed, selected = set(), set()
    while pending:
        name = pending.pop()
        if name in needed:
            continue
        needed.add(name)
        index = producers.get(name)
        if index is not None:
            selected.add(index)
            pending.extend(nodes[index].input)
    graph = helper.make_graph([n for index, n in enumerate(nodes) if index in selected],
                              "dequantized_decoder_probe", list(model.graph.input), list(model.graph.output),
                              [i for name, i in initializers.items() if name in needed])
    model.graph.CopyFrom(graph)
    onnx.checker.check_model(model)
    onnx.save(model, target)
    print(json.dumps({"diagnostic_only": True, "replaced_convolutions": len(replacements),
                      "path": str(target)}))


def probe(args):
    import numpy as np
    import onnxruntime as ort
    folder = pathlib.Path(args.output)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    reference = ort.InferenceSession(str(folder / "reference.onnx"), options, providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(str(folder / args.decoder), options, providers=["CPUExecutionProvider"])
    encoder = ort.InferenceSession(str(folder / "encoder.onnx"), options, providers=["CPUExecutionProvider"])
    inputs = {"input": np.array([[1] + list(range(2, 22)) * 3 + [2]], dtype=np.int64),
              "input_lengths": np.array([62], dtype=np.int64),
              "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32)}
    if args.tokens:
        ids = json.loads(pathlib.Path(args.tokens).read_text(encoding="utf-8"))["ids"]
        inputs["input"] = np.array([ids], dtype=np.int64)
        inputs["input_lengths"] = np.array([len(ids)], dtype=np.int64)
    artifacts = pathlib.Path(args.artifacts) if args.artifacts else None
    if artifacts:
        artifacts.mkdir(parents=True, exist_ok=True)
        assert not any(artifacts.iterdir()), "Use a new empty artifact directory"
        (artifacts / "tokens.txt").write_text(" ".join(map(str, inputs["input"][0])), encoding="ascii")
    def save_audio(name, samples):
        if artifacts:
            import wave
            rate = int(reference.get_modelmeta().custom_metadata_map["sample_rate"])
            with wave.open(str(artifacts / name), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(rate)
                wav.writeframes((np.clip(samples.reshape(-1), -1, 1)*32767).astype("<i2").tobytes())
    begin = time.perf_counter()
    audio, latent = reference.run(None, inputs)
    assert np.isfinite(audio).all() and np.isfinite(latent).all()
    full_ms = (time.perf_counter() - begin) * 1000
    save_audio("reference.wav", audio)
    begin = time.perf_counter()
    encoded = encoder.run(None, inputs)[0]
    encoder_ms = (time.perf_counter() - begin) * 1000
    assert encoded.shape[1] == latent.shape[1] == 192
    feed = decoder.get_inputs()[0].name
    begin = time.perf_counter()
    decoded = decoder.run(None, {feed: latent})[0]
    decoder_ms = (time.perf_counter() - begin) * 1000
    stride = audio.size // latent.shape[-1]
    assert audio.size == stride * latent.shape[-1]
    results = []
    for block_frames, overlap in ((32, 8), (32, 16), (32, 32), (64, 8), (64, 16), (128, 8)):
        pieces = []
        begin = time.perf_counter()
        first_ms = None
        for start in range(0, latent.shape[-1], block_frames):
            end = min(start + block_frames, latent.shape[-1])
            left, right = max(0, start - overlap), min(latent.shape[-1], end + overlap)
            block = decoder.run(None, {feed: latent[:, :, left:right]})[0].reshape(-1)
            pieces.append(block[(start-left)*stride:(end-left)*stride])
            if first_ms is None:
                first_ms = (time.perf_counter() - begin) * 1000
        joined = np.concatenate(pieces)
        assert np.isfinite(joined).all()
        decoder_total_ms = (time.perf_counter()-begin)*1000
        save_audio(f"chunk-{block_frames}-{overlap}.wav", joined)
        assert joined.size == audio.size, "Lost or repeated samples"
        diff = joined - audio.reshape(-1)
        decoder_diff = joined - decoded.reshape(-1)
        results.append({"block_frames": block_frames, "overlap_frames": overlap, "first_decoder_block_ms": first_ms,
                        "total_decoder_ms": decoder_total_ms,
                        "max_abs_error": float(np.max(np.abs(diff))),
                        "rms_error": float(np.sqrt(np.mean(diff**2))),
                        "max_vs_full_decoder": float(np.max(np.abs(decoder_diff))),
                        "rms_vs_full_decoder": float(np.sqrt(np.mean(decoder_diff**2)))})
    print(json.dumps({"synthetic_input_only": not bool(args.tokens), "desktop_only": True,
                      "latent_shape": list(latent.shape), "samples_per_frame": stride,
                      "full_ms": full_ms, "encoder_ms": encoder_ms, "decoder_ms": decoder_ms,
                      "split_full_max_error": float(np.max(np.abs(decoded-audio))),
                      "chunked": results}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["export", "capture", "probe", "float-decoder"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--archive")
    parser.add_argument("--text")
    parser.add_argument("--tokens")
    parser.add_argument("--artifacts")
    parser.add_argument("--decoder", default="decoder.onnx")
    parser.add_argument("--member", default="files/models/piper/zh/zh_CN-xiao_ya-medium.onnx")
    args = parser.parse_args()
    {"export": export, "capture": capture, "probe": probe, "float-decoder": float_decoder}[args.mode](args)
