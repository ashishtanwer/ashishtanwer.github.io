#!/usr/bin/env python3
"""Capture a separate, real 128D RoPE example from pinned Qwen3-0.6B.

This is not a Granite head or an extension of the Granite forward pass. It
observes block 0, query head 0 and unique key/value head 0 of Qwen3-0.6B.
Q/K are observed after Qwen's per-head RMSNorm, at the actual RoPE boundary.
Transparent wrappers call the original implementation and return its original
tensors unchanged. No existing Granite trace files are modified.

    python scripts/capture_rope_example.py --cache-dir /tmp/moe-hf-cache
    python scripts/capture_rope_example.py --validate
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import wraps
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform


MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
CONFIG_SHA256 = "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"
WEIGHTS_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
PROMPT = "The capital of France is"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rounded(value):
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return round(value, 9)


def validate_example(document):
    """Independently reconstruct the serialized rotations using Python floats."""
    assert document["schemaVersion"] == 1 and document["kind"] == "rope-example"
    model = document["model"]
    assert model["id"] == MODEL_ID and model["revision"] == MODEL_REVISION
    assert model["headSize"] == 128
    assert model["attentionHeads"] == 16 and model["keyValueHeads"] == 8
    assert model["ropeTheta"] == 1_000_000 and model["ropeScaling"] is None
    assert model["attentionMultiplier"] == 128 ** -0.5
    assert document["scope"] == {"layer": 0, "queryHead": 0, "keyValueHead": 0}
    example = document["example"]
    assert example["id"] == "qwen3-128" and example["prompt"] == PROMPT
    tokens = example["tokens"]
    assert "".join(token["text"] for token in tokens) == PROMPT
    count = len(tokens)
    assert document["positions"] == list(range(count))
    assert [token["index"] for token in tokens] == list(range(count))
    half = model["headSize"] // 2
    for name in ("queryBefore", "queryAfter", "keyBefore", "keyAfter", "value"):
        assert len(document[name]) == count
        assert all(len(row) == 128 and all(math.isfinite(value) for value in row) for row in document[name])
    for name in ("cos", "sin"):
        assert len(document["rotary"][name]) == count
        assert all(len(row) == half and all(math.isfinite(value) for value in row)
                   for row in document["rotary"][name])
    maximum_error = 0.0
    maximum_relative_error = 0.0
    maximum_pair_norm_error = 0.0
    pairs = 0
    for token in range(count):
        for coordinate in range(half):
            cos = document["rotary"]["cos"][token][coordinate]
            sin = document["rotary"]["sin"][token][coordinate]
            assert math.isclose(cos * cos + sin * sin, 1, abs_tol=2e-6)
            angle = document["positions"][token] / model["ropeTheta"] ** (2 * coordinate / model["headSize"])
            assert math.isclose(cos, math.cos(angle), abs_tol=2e-6)
            assert math.isclose(sin, math.sin(angle), abs_tol=2e-6)
            for kind in ("query", "key"):
                before = document[f"{kind}Before"][token]
                after = document[f"{kind}After"][token]
                first, second = before[coordinate], before[coordinate + half]
                expected = (first * cos - second * sin, second * cos + first * sin)
                actual = (after[coordinate], after[coordinate + half])
                pairs += 1
                for computed, observed in zip(expected, actual):
                    error = abs(observed - computed)
                    maximum_error = max(maximum_error, error)
                    maximum_relative_error = max(maximum_relative_error, error / max(abs(observed), 1e-5))
                    assert math.isclose(observed, computed, rel_tol=2e-6, abs_tol=1e-5)
                before_norm = math.hypot(first, second)
                after_norm = math.hypot(*actual)
                maximum_pair_norm_error = max(maximum_pair_norm_error, abs(after_norm - before_norm))
                assert math.isclose(before_norm, after_norm, rel_tol=2e-6, abs_tol=1e-5)
    assert document["queryBefore"][0] == document["queryAfter"][0]
    assert document["keyBefore"][0] == document["keyAfter"][0]
    capture_info = document["capture"]
    assert capture_info["queryInputNormalization"] == "q_norm"
    assert capture_info["keyInputNormalization"] == "k_norm"
    assert capture_info["actualNormalizationOutputsMatchRotaryInputs"] is True
    assert capture_info["actualValueUnchangedByRotation"] is True
    assert capture_info["actualGroupedKeyValueRepetitionExact"] is True
    assert capture_info["hookedVsUnhookedMaxLogitError"] == 0
    assert capture_info["configSha256"] == CONFIG_SHA256
    assert capture_info["modelFiles"][0]["sha256"] == WEIGHTS_SHA256
    return {"passed": True, "headSize": 128, "pairsChecked": pairs,
            "ropeMaxAbsError": maximum_error, "ropeMaxRelativeError": maximum_relative_error,
            "maxPairNormError": maximum_pair_norm_error, "positionZeroUnchanged": True,
            "frequenciesMatchConfiguredTheta": True}


def capture(args):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3 import modeling_qwen3 as implementation

    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    snapshot = Path(snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        allow_patterns=["*.json", "*.safetensors", "merges.txt", "LICENSE"],
    ))
    assert sha256_file(snapshot / "config.json") == CONFIG_SHA256
    assert sha256_file(snapshot / "model.safetensors") == WEIGHTS_SHA256
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, attn_implementation="eager",
    ).eval()
    config = model.config
    attention = model.model.layers[0].self_attn
    assert config.model_type == "qwen3" and config.head_dim == 128
    assert config.num_attention_heads == 16 and config.num_key_value_heads == 8
    assert config.num_hidden_layers == 28 and config.hidden_size == 1024
    assert config.rope_theta == 1_000_000 and config.rope_scaling is None
    assert attention.scaling == config.head_dim ** -0.5
    encoded = tokenizer(PROMPT, return_tensors="pt", add_special_tokens=False)
    count = encoded["input_ids"].shape[1]
    cache = {}
    active = False
    original_rotary = implementation.apply_rotary_pos_emb
    original_eager = implementation.eager_attention_forward
    original_repeat = implementation.repeat_kv

    def clone(tensor):
        return tensor.detach().cpu().clone()

    def before_attention(_module, _args, kwargs):
        nonlocal active
        assert not active
        active = True
        cache["positions"] = clone(kwargs["position_ids"])

    def after_attention(_module, _args, _kwargs, _output):
        nonlocal active
        active = False

    def observe_module(name):
        def hook(_module, _args, output):
            cache[name] = clone(output)
        return hook

    @wraps(original_rotary)
    def observe_rotary(query, key, cos, sin, *positional, **keyword):
        result = original_rotary(query, key, cos, sin, *positional, **keyword)
        if active:
            assert "queryBefore" not in cache
            cache.update(queryBefore=clone(query), keyBefore=clone(key),
                         queryAfter=clone(result[0]), keyAfter=clone(result[1]),
                         cos=clone(cos), sin=clone(sin), repetitions=[])
        return result

    @wraps(original_repeat)
    def observe_repeat(hidden, groups):
        result = original_repeat(hidden, groups)
        if active:
            cache["repetitions"].append((clone(hidden), clone(result), groups))
        return result

    @wraps(original_eager)
    def observe_eager(module, query, key, value, *positional, **keyword):
        if active:
            assert module is attention
            cache.update(attentionQuery=clone(query), attentionKey=clone(key), value=clone(value))
        return original_eager(module, query, key, value, *positional, **keyword)

    handles = [
        attention.register_forward_pre_hook(before_attention, with_kwargs=True),
        attention.q_norm.register_forward_hook(observe_module("queryNormalized")),
        attention.k_norm.register_forward_hook(observe_module("keyNormalized")),
        attention.v_proj.register_forward_hook(observe_module("valueProjection")),
        attention.register_forward_hook(after_attention, with_kwargs=True),
    ]
    implementation.apply_rotary_pos_emb = observe_rotary
    implementation.eager_attention_forward = observe_eager
    implementation.repeat_kv = observe_repeat
    try:
        print(f"Capturing {MODEL_ID} block 0 Q0/KV0, {count} tokens, actual 128D RoPE", flush=True)
        with torch.inference_mode():
            observed = model(**encoded, use_cache=False, return_dict=True)
    finally:
        implementation.apply_rotary_pos_emb = original_rotary
        implementation.eager_attention_forward = original_eager
        implementation.repeat_kv = original_repeat
        for handle in handles:
            handle.remove()
    with torch.inference_mode():
        unobserved = model(**encoded, use_cache=False, return_dict=True)
    torch.testing.assert_close(observed.logits, unobserved.logits, rtol=0, atol=0)
    logit_error = (observed.logits - unobserved.logits).abs().max().item()
    assert cache["positions"][0].tolist() == list(range(count))
    for kind in ("query", "key"):
        torch.testing.assert_close(cache[f"{kind}Normalized"].transpose(1, 2), cache[f"{kind}Before"], rtol=0, atol=0)
        torch.testing.assert_close(cache[f"{kind}After"], cache[f"attention{kind.title()}"], rtol=0, atol=0)
    value_heads = cache["valueProjection"].view(1, count, 8, 128).transpose(1, 2)
    torch.testing.assert_close(value_heads, cache["value"], rtol=0, atol=0)
    assert len(cache["repetitions"]) == 2
    for (before, after, groups), expected in zip(cache["repetitions"], (cache["keyAfter"], cache["value"])):
        assert groups == 2
        torch.testing.assert_close(before, expected, rtol=0, atol=0)
        for query_head in range(16):
            torch.testing.assert_close(after[:, query_head], expected[:, query_head // 2], rtol=0, atol=0)
    for name in ("cos", "sin"):
        torch.testing.assert_close(cache[name][..., :64], cache[name][..., 64:], rtol=0, atol=0)
    for kind in ("query", "key"):
        before = cache[f"{kind}Before"]
        first = before[..., :64] * cache["cos"][:, None, :, :64] - before[..., 64:] * cache["sin"][:, None, :, :64]
        second = before[..., 64:] * cache["cos"][:, None, :, :64] + before[..., :64] * cache["sin"][:, None, :, :64]
        torch.testing.assert_close(torch.cat((first, second), dim=-1), cache[f"{kind}After"], rtol=0, atol=0)
    tokens = [{"index": index, "id": token_id,
               "text": tokenizer.decode([token_id], clean_up_tokenization_spaces=False)}
              for index, token_id in enumerate(encoded["input_ids"][0].tolist())]
    document = {
        "schemaVersion": 1, "kind": "rope-example",
        "model": {
            "id": MODEL_ID, "name": "Qwen3 0.6B", "url": f"https://huggingface.co/{MODEL_ID}",
            "revision": MODEL_REVISION, "headSize": config.head_dim,
            "attentionHeads": config.num_attention_heads, "keyValueHeads": config.num_key_value_heads,
            "hiddenSize": config.hidden_size, "numLayers": config.num_hidden_layers,
            "ropeTheta": config.rope_theta, "ropeScaling": config.rope_scaling,
            "attentionMultiplier": attention.scaling, "license": "Apache-2.0",
        },
        "capture": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/capture_rope_example.py", "scriptSha256": sha256_file(Path(__file__)),
            "device": "cpu", "dtype": "float32", "python": platform.python_version(),
            "packages": {name: metadata.version(name) for name in
                         ("torch", "transformers", "safetensors", "huggingface-hub", "tokenizers")},
            "modelSourceSha256": sha256_file(implementation.__file__),
            "modelSource": "transformers/models/qwen3/modeling_qwen3.py",
            "modelSourceUrl": "https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py",
            "configSha256": CONFIG_SHA256,
            "modelFiles": [{"name": "model.safetensors", "bytes": (snapshot / "model.safetensors").stat().st_size,
                            "sha256": WEIGHTS_SHA256}],
            "tokenizerFiles": [{"name": name, "sha256": sha256_file(snapshot / name)} for name in
                               ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")],
            "hookedVsUnhookedMaxLogitError": logit_error,
            "logitsSha256": hashlib.sha256(observed.logits.cpu().numpy().tobytes()).hexdigest(),
            "actualValueUnchangedByRotation": True, "actualGroupedKeyValueRepetitionExact": True,
            "actualNormalizationOutputsMatchRotaryInputs": True,
            "queryInputNormalization": "q_norm", "keyInputNormalization": "k_norm",
            "notes": [
                "Separate example from Qwen3-0.6B; this is not Granite data and does not alter the Granite model diagram or recorded traces.",
                "The original installed apply_rotary_pos_emb is called by a transparent wrapper; queryBefore/keyBefore and queryAfter/keyAfter are its actual inputs and outputs, not solely a separate reconstruction.",
                "Qwen3 applies per-head q_norm and k_norm before RoPE. Their observed outputs, after the actual head transpose, are verified exactly equal to the recorded RoPE inputs.",
                "All 128 coordinates of block 0 query head 0 and unique key/value head 0 are recorded for every prompt token. Half-split pairs are (i,i+64), for i=0 through 63.",
                "rotary.cos/sin contains the first 64 coordinates of actual forward-pass rotary tensors; duplicated second halves are verified identical. ropeScaling is null and ropeTheta is 1000000.",
                "The actual V input to eager attention is verified exactly equal to the V projection reshaped into heads. RoPE leaves V unchanged. Actual grouped key/value repetition is also checked exactly for all 16 query heads.",
                "A second model execution after removing every observation hook/wrapper has exactly identical logits for every prompt position and all 151936 vocabulary entries.",
                "The pretrained bfloat16 weights are loaded as float32 for this CPU capture. Arrays are rounded to nine decimal places; independent validation reconstructs every stored pair with rtol=2e-6 and atol=1e-5.",
                "Attention scales QK by 1/sqrt(128), distinct from Granite's 1/64. The example captures a direct text prompt without a chat template, BOS token, or generated continuation.",
            ],
        },
        "example": {"id": "qwen3-128", "prompt": PROMPT, "tokens": tokens},
        "scope": {"layer": 0, "queryHead": 0, "keyValueHead": 0},
        "positions": cache["positions"][0].tolist(),
        "rotary": {name: rounded(cache[name][0, :, :64].tolist()) for name in ("cos", "sin")},
        **{name: rounded(cache[name][0, 0].tolist()) for name in
           ("queryBefore", "queryAfter", "keyBefore", "keyAfter", "value")},
    }
    document["validation"] = validate_example(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Wrote {args.output} ({args.output.stat().st_size:,} bytes): {document['validation']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("static/traces/rope-example-128.json"))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--validate", action="store_true", help="Validate the saved example using standard-library Python")
    args = parser.parse_args()
    if args.validate:
        document = json.loads(args.output.read_text())
        assert document["capture"]["scriptSha256"] == sha256_file(Path(__file__))
        print(json.dumps(validate_example(document)))
    else:
        capture(args)


if __name__ == "__main__":
    main()
