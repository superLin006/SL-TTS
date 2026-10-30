"""Generate distributable Piper streaming assets; no device-specific files are required.

Use --archive for the existing combined device archive, or download the original
model packs in CI. Source weights are verified against the pinned SHA-256s.
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import tarfile
import tempfile
import urllib.request
from types import SimpleNamespace
from probe_vits_streaming import export, float_decoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive")
    parser.add_argument("--output", default="app/src/main/assets/streaming-piper-v1")
    args = parser.parse_args()
    pins = json.loads(pathlib.Path(__file__).with_name("streaming_model_sources.json").read_text(encoding="utf-8"))
    root = pathlib.Path(args.output)
    for lang, source in pins.items():
        with tempfile.TemporaryDirectory(prefix="livebabel-stream-build-") as temporary:
            tmp = pathlib.Path(temporary)
            archive = pathlib.Path(args.archive) if args.archive else tmp / "source.tar.bz2"
            if not args.archive:
                urllib.request.urlretrieve(source["url"], archive)
            with tarfile.open(archive) as tar:
                candidates = [m.name for m in tar.getmembers() if m.isfile() and pathlib.PurePosixPath(m.name).name == source["model"]]
                assert len(candidates) == 1, "Ambiguous source model"
                member = candidates[0]
                original = tar.extractfile(member).read()
                assert hashlib.sha256(original).hexdigest() == source["sha256"], "Unexpected source weights"
                prefix = str(pathlib.PurePosixPath(member).parent)
                card = tar.extractfile(prefix + "/MODEL_CARD").read()
            output = tmp / "export"
            export(SimpleNamespace(archive=str(archive), member=member, output=str(output)))
            float_decoder(SimpleNamespace(output=str(output)))
            destination = root / lang
            destination.mkdir(parents=True, exist_ok=True)
            files = {}
            for name in ("encoder.onnx", "decoder-float.onnx", "frontend-ids-only.onnx"):
                shutil.copyfile(output / name, destination / name)
                files[name] = hashlib.sha256((destination / name).read_bytes()).hexdigest()
            (destination / "MODEL_CARD").write_bytes(card)
            (destination / "manifest.json").write_text(json.dumps({"version": 1,
                "source_sha256": source["sha256"], "sample_rate": source["sample_rate"], "files": files}, indent=2), encoding="utf-8")
            print(f"Prepared {lang}: {destination}")


if __name__ == "__main__":
    main()
