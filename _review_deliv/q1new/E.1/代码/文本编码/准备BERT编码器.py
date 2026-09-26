"""Expose the final 768-dimensional encoder state of the official BERT ONNX model.

The downloaded ONNX file is BertForMaskedLM and exposes vocabulary logits only.
This reproducible transformation removes the masked-language-model head; it does
not alter any encoder weights. Run once before extracting word features.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "运行环境" / "onnx_packages"))

import onnx


SOURCE = ROOT / "模型" / "BERT_base_uncased" / "model.onnx"
DESTINATION = ROOT / "模型" / "BERT_base_uncased" / "model_encoder_768.onnx"
DETAILS = ROOT / "模型" / "BERT_base_uncased" / "model_encoder_768.json"
SOURCE_SHA256 = "44d7a2896d341c51fb1eba89aea3a590e6af0ce33e25481136f7eeecb62e5f7f"
HIDDEN_NAME = "/bert/encoder/layer.11/output/LayerNorm/Add_1_output_0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if sha256(SOURCE) != SOURCE_SHA256:
        raise ValueError("Official BERT ONNX checksum mismatch")
    model = onnx.load(str(SOURCE), load_external_data=False)
    positions = [i for i, node in enumerate(model.graph.node) if HIDDEN_NAME in node.output]
    if len(positions) != 1 or model.graph.node[positions[0]].op_type != "Add":
        raise ValueError("Cannot uniquely locate the final BERT encoder LayerNorm")
    cut = positions[0]
    del model.graph.node[cut + 1:]
    del model.graph.output[:]
    model.graph.output.append(onnx.helper.make_tensor_value_info(
        HIDDEN_NAME, onnx.TensorProto.FLOAT, ["batch_size", "sequence_length", 768]
    ))
    used = {name for node in model.graph.node for name in node.input}
    for i in range(len(model.graph.initializer) - 1, -1, -1):
        if model.graph.initializer[i].name not in used:
            del model.graph.initializer[i]
    model.graph.value_info.clear()
    model.metadata_props.add(key="derived_from", value="google-bert/bert-base-uncased model.onnx")
    model.metadata_props.add(key="source_sha256", value=SOURCE_SHA256)
    onnx.save(model, str(DESTINATION))
    onnx.checker.check_model(str(DESTINATION))
    info = {
        "source_repo": "google-bert/bert-base-uncased",
        "source_revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
        "source_file": SOURCE.name,
        "source_sha256": SOURCE_SHA256,
        "encoder_output_name": HIDDEN_NAME,
        "encoder_file": DESTINATION.name,
        "encoder_sha256": sha256(DESTINATION),
        "encoder_dimensions": 768,
        "model_transform": "Remove masked-language-model head after final encoder LayerNorm; keep encoder weights unchanged",
        "onnx_version": onnx.__version__,
    }
    DETAILS.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
