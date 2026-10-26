#!/usr/bin/env python3
"""Record actual head splitting, RoPE, and grouped-query sharing operations.

Transparent wrappers call the installed Granite implementation's original
apply_rotary_pos_emb and repeat_kv functions and record their actual inputs
and outputs. The inference path receives the original return values unchanged.
The main trace and expert-operation sidecars are never modified.

    python scripts/capture_attention_operations.py --cache-dir /tmp/moe-hf-cache
    python scripts/capture_attention_operations.py --validate
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import wraps
from importlib import metadata
import json
import math
from pathlib import Path
import platform

from capture_moe_traces import MODEL_ID, MODEL_REVISION, rounded, sha256_file


def validate_sidecar(sidecar, main):
    assert sidecar["schemaVersion"] == 1 and sidecar["kind"] == "attention-operations"
    model = sidecar["model"]
    assert model["id"] == main["model"]["id"] and model["revision"] == main["model"]["revision"]
    assert model["attentionHeads"] == main["model"]["attentionHeads"]
    assert model["keyValueHeads"] == main["model"]["keyValueHeads"]
    assert model["headSize"] == main["model"]["attentionHeadSize"]
    assert model["attentionMultiplier"] == main["model"]["attentionMultiplier"]
    example = next(example for example in main["examples"] if example["id"] == sidecar["example"]["id"])
    assert sidecar["example"]["prompt"] == example["prompt"]
    assert sidecar["example"]["tokens"] == example["tokens"]
    count = len(example["tokens"])
    assert sidecar["positions"] == list(range(count))
    assert len(sidecar["layers"]) == len(example["layers"])
    half = model["headSize"] // 2
    assert len(sidecar["rotary"]["cos"]) == len(sidecar["rotary"]["sin"]) == count
    for cos, sin in zip(sidecar["rotary"]["cos"], sidecar["rotary"]["sin"]):
        assert len(cos) == len(sin) == half
        assert all(math.isfinite(value) for value in cos + sin)
        assert all(math.isclose(c*c+s*s, 1, rel_tol=2e-6, abs_tol=2e-6) for c, s in zip(cos, sin))
    group_size = model["attentionHeads"] // model["keyValueHeads"]
    maximum_rotation_error = 0.0
    maximum_score_error = 0.0
    rotated_pairs = 0
    score_count = 0
    for layer_index, layer in enumerate(sidecar["layers"]):
        assert layer["index"] == layer_index
        original = example["layers"][layer_index]
        for field in ("queryBefore", "queryAfter", "keyBefore", "keyAfter", "value"):
            heads = model["attentionHeads"] if field.startswith("query") else model["keyValueHeads"]
            assert len(layer[field]) == heads
            for matrix in layer[field]:
                assert len(matrix) == count
                assert all(len(row) == model["headSize"] and all(math.isfinite(value) for value in row) for row in matrix)
        for kind in ("query", "key"):
            for before, after in zip(layer[f"{kind}Before"], layer[f"{kind}After"]):
                for token_index in range(count):
                    for coordinate in range(half):
                        rotated_pairs += 1
                        cos = sidecar["rotary"]["cos"][token_index][coordinate]
                        sin = sidecar["rotary"]["sin"][token_index][coordinate]
                        first, second = before[token_index][coordinate], before[token_index][coordinate + half]
                        expected_first = first * cos - second * sin
                        expected_second = second * cos + first * sin
                        for actual, expected in ((after[token_index][coordinate], expected_first),
                                                 (after[token_index][coordinate + half], expected_second)):
                            maximum_rotation_error = max(maximum_rotation_error, abs(actual - expected))
                            assert math.isclose(actual, expected, rel_tol=2e-6, abs_tol=1e-5)
                assert before[0] == after[0], "Position zero must leave all coordinates unchanged"
        for head in range(model["attentionHeads"]):
            kv_head = head // group_size
            for token_index in range(count):
                assert layer["queryAfter"][head][token_index][:8] == original["queryPreview"][head][token_index]
                assert layer["keyAfter"][kv_head][token_index][:8] == original["keyPreview"][head][token_index]
                assert layer["value"][kv_head][token_index][:8] == original["valuePreview"][head][token_index]
            for query in range(count):
                for key in range(count):
                    score_count += 1
                    terms = [a*b for a, b in zip(layer["queryAfter"][head][query], layer["keyAfter"][kv_head][key])]
                    reconstructed = sum(terms) * model["attentionMultiplier"]
                    actual = example["attentionScores"][layer_index][head][query][key]
                    error = abs(actual - reconstructed)
                    maximum_score_error = max(maximum_score_error, error)
                    assert error <= 1e-5 + 2e-6 * sum(abs(term) for term in terms) * model["attentionMultiplier"]
    assert sidecar["capture"]["matchesMainVocabularyLogits"] is True
    assert sidecar["capture"]["actualValueUnchangedByRotation"] is True
    assert sidecar["capture"]["actualGroupedKeyValueRepetitionExact"] is True
    assert sidecar["capture"]["projectionHeadReshapeExact"] is True
    return {"passed": True, "rotatedCoordinatePairs": rotated_pairs, "reconstructedAttentionScores": score_count,
            "maxRotationReconstructionError": maximum_rotation_error,
            "maxAttentionScoreReconstructionError": maximum_score_error,
            "positionZeroUnchanged": True, "matchesMainPostRotationPreviews": True,
            "matchesMainRepeatedKeyValuePreviews": True}


def capture(args):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.granitemoe import modeling_granitemoe as implementation

    main_bytes = args.bundle.read_bytes()
    main_hash = sha256_file(args.bundle)
    main = json.loads(main_bytes)
    assert main["model"]["id"] == MODEL_ID and main["model"]["revision"] == MODEL_REVISION
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, cache_dir=args.cache_dir,
                                      local_files_only=args.local_files_only,
                                      allow_patterns=["*.json", "*.safetensors"]))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, attn_implementation="eager",
    ).eval()
    config = model.config
    head_size = config.hidden_size // config.num_attention_heads
    group_size = config.num_attention_heads // config.num_key_value_heads
    for filename in main["capture"]["modelFiles"]:
        assert sha256_file(snapshot / filename["name"]) == filename["sha256"]
    cache = {}
    layer_records = {}
    shared = {}
    handles = []
    active_layer = None
    original_rotary = implementation.apply_rotary_pos_emb
    original_repeat = implementation.repeat_kv

    def snapshot_tensor(tensor):
        return tensor.detach().cpu().clone()

    @wraps(original_rotary)
    def observe_rotary(query, key, cos, sin, *positional, **keyword):
        assert active_layer is not None, "Each actual rotary call must belong to an observed attention layer"
        result = original_rotary(query, key, cos, sin, *positional, **keyword)
        cache[active_layer].update(
            queryBefore=snapshot_tensor(query), keyBefore=snapshot_tensor(key),
            queryAfter=snapshot_tensor(result[0]), keyAfter=snapshot_tensor(result[1]),
            cos=snapshot_tensor(cos), sin=snapshot_tensor(sin), repetitions=[],
        )
        return result

    @wraps(original_repeat)
    def observe_repeat(hidden, groups):
        assert active_layer is not None
        result = original_repeat(hidden, groups)
        cache[active_layer]["repetitions"].append((snapshot_tensor(hidden), snapshot_tensor(result), groups))
        return result

    def before_attention(index):
        def hook(_module, _args, kwargs):
            nonlocal active_layer
            assert active_layer is None
            active_layer = index
            cache[index] = {"positions": snapshot_tensor(kwargs["position_ids"])}
        return hook

    def projection_hook(index, kind):
        def hook(_module, _args, output):
            cache[index][f"{kind}Projection"] = snapshot_tensor(output)
        return hook

    def after_attention(index):
        def hook(module, _args, _kwargs, output):
            nonlocal active_layer
            data = cache.pop(index)
            assert active_layer == index
            active_layer = None
            batch, length, _ = data["queryProjection"].shape
            assert batch == 1
            query_heads = data["queryProjection"].view(batch, length, config.num_attention_heads, head_size).transpose(1, 2)
            key_heads = data["keyProjection"].view(batch, length, config.num_key_value_heads, head_size).transpose(1, 2)
            value_heads = data["valueProjection"].view(batch, length, config.num_key_value_heads, head_size).transpose(1, 2)
            torch.testing.assert_close(query_heads, data["queryBefore"], rtol=0, atol=0)
            torch.testing.assert_close(key_heads, data["keyBefore"], rtol=0, atol=0)
            assert len(data["repetitions"]) == 2
            key_input, repeated_key, key_groups = data["repetitions"][0]
            value_input, repeated_value, value_groups = data["repetitions"][1]
            assert key_groups == value_groups == group_size
            torch.testing.assert_close(key_input, data["keyAfter"], rtol=0, atol=0)
            torch.testing.assert_close(value_input, value_heads, rtol=0, atol=0)
            for query_head in range(config.num_attention_heads):
                unique_head = query_head // group_size
                torch.testing.assert_close(repeated_key[:, query_head], data["keyAfter"][:, unique_head], rtol=0, atol=0)
                torch.testing.assert_close(repeated_value[:, query_head], value_heads[:, unique_head], rtol=0, atol=0)
            half = head_size // 2
            torch.testing.assert_close(data["cos"][..., :half], data["cos"][..., half:], rtol=0, atol=0)
            torch.testing.assert_close(data["sin"][..., :half], data["sin"][..., half:], rtol=0, atol=0)
            for name in ("query", "key"):
                before = data[f"{name}Before"]
                first = before[..., :half] * data["cos"][:, None, :, :half] - before[..., half:] * data["sin"][:, None, :, :half]
                second = before[..., half:] * data["cos"][:, None, :, :half] + before[..., :half] * data["sin"][:, None, :, :half]
                torch.testing.assert_close(torch.cat((first, second), dim=-1), data[f"{name}After"], rtol=0, atol=0)
            scores = data["queryAfter"] @ repeated_key.transpose(2, 3) * module.scaling
            mask = torch.ones((length, length), dtype=torch.bool).triu(1)
            probabilities = scores.masked_fill(mask, torch.finfo(scores.dtype).min).softmax(dim=-1)
            torch.testing.assert_close(probabilities, output[1].detach().cpu(), rtol=0, atol=0)
            shared_values = {"positions": data["positions"][0].tolist(),
                             "cos": rounded(data["cos"][0, :, :half].tolist()),
                             "sin": rounded(data["sin"][0, :, :half].tolist())}
            if shared:
                assert shared == shared_values, "Every layer must use the actual shared position encoding"
            else:
                shared.update(shared_values)
            layer_records[index] = {
                "index": index,
                **{name: rounded(data[name][0].tolist()) for name in
                   ("queryBefore", "keyBefore", "queryAfter", "keyAfter")},
                "value": rounded(value_heads[0].tolist()),
            }
        return hook

    for index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        handles.extend([
            attention.register_forward_pre_hook(before_attention(index), with_kwargs=True),
            attention.q_proj.register_forward_hook(projection_hook(index, "query")),
            attention.k_proj.register_forward_hook(projection_hook(index, "key")),
            attention.v_proj.register_forward_hook(projection_hook(index, "value")),
            attention.register_forward_hook(after_attention(index), with_kwargs=True),
        ])
    implementation.apply_rotary_pos_emb = observe_rotary
    implementation.repeat_kv = observe_repeat
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for example in main["examples"]:
            print(f"Capturing actual head split / RoPE / GQA: {example['id']}", flush=True)
            layer_records.clear()
            shared.clear()
            encoded = tokenizer(example["prompt"], return_tensors="pt", add_special_tokens=False)
            assert encoded["input_ids"][0].tolist() == [token["id"] for token in example["tokens"]]
            with torch.inference_mode():
                output = model(**encoded, use_cache=False, output_attentions=True, output_router_logits=True, return_dict=True)
            assert rounded(output.logits[0, -1].tolist()) == example["nextTokenLogits"]
            assert [rounded(matrix[0].tolist()) for matrix in output.attentions] == example["attention"]
            for index, logits in enumerate(output.router_logits):
                assert rounded(logits.tolist()) == [token["routerLogits"] for token in example["layers"][index]["tokens"]]
            sidecar = {
                "schemaVersion": 1, "kind": "attention-operations",
                "model": {
                    "id": MODEL_ID, "revision": MODEL_REVISION,
                    "attentionHeads": config.num_attention_heads, "keyValueHeads": config.num_key_value_heads,
                    "headSize": head_size, "ropeTheta": config.rope_theta,
                    "attentionMultiplier": config.attention_multiplier,
                },
                "capture": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "script": "scripts/capture_attention_operations.py", "scriptSha256": sha256_file(Path(__file__)),
                    "mainBundleSha256": main_hash, "mainCaptureScriptSha256": main["capture"]["scriptSha256"],
                    "device": "cpu", "dtype": "float32", "python": platform.python_version(),
                    "packages": {name: metadata.version(name) for name in
                                 ("torch", "transformers", "safetensors", "huggingface-hub")},
                    "matchesMainVocabularyLogits": True, "actualValueUnchangedByRotation": True,
                    "actualGroupedKeyValueRepetitionExact": True, "projectionHeadReshapeExact": True,
                    "notes": [
                        "Transparent wrappers call the original installed apply_rotary_pos_emb and repeat_kv functions and observe their actual inputs/outputs. The model receives the original returned tensors unchanged; post-RoPE data is not generated solely by a separate reconstruction.",
                        "Before arrays are the actual Q/K projection outputs reshaped into 16 query heads and 8 unique key heads. value contains 8 actual unique V projection heads. Each complete head has 64 coordinates.",
                        "Projection reshaping is checked exactly against the actual inputs to apply_rotary_pos_emb. The actual input to repeat_kv for V is checked exactly against the observed V projection output, proving V is unchanged by RoPE.",
                        "The actual repeated K/V outputs are checked coordinate by coordinate: query head h shares unique key/value head floor(h/2). Stored K/V arrays retain only the 8 unique heads.",
                        "RoPE pairs coordinate i with i+32. First'=first*cos-second*sin; second'=second*cos+first*sin. All 32 pairs are recorded and independently verified against the actual original-function outputs.",
                        "rotary.cos/sin stores the first 32 coordinates of actual forward-pass rotary tensors; their second halves are verified exactly identical. Every layer is verified to use the same captured position encoding.",
                        "All router logits, all 49152 final-token vocabulary logits, and all returned attention probabilities exactly match the unchanged main bundle after its documented nine-decimal serialization.",
                        "Arrays are rounded to 9 decimal places. Validators reconstruct every full-coordinate rotation and every scaled QK attention score with floating-point tolerances and verify all main post-RoPE/QKV previews.",
                    ],
                },
                "example": {key: example[key] for key in ("id", "prompt", "tokens")},
                "positions": shared["positions"],
                "rotary": {"cos": shared["cos"], "sin": shared["sin"]},
                "layers": [layer_records[index] for index in range(config.num_hidden_layers)],
            }
            sidecar["validation"] = validate_sidecar(sidecar, main)
            path = args.output_dir / f"attention-operations-{example['id']}.json"
            serialized = json.dumps(sidecar, ensure_ascii=False, separators=(",", ":")) + "\n"
            assert len(serialized.encode()) < 25_000_000
            path.write_text(serialized)
            print(f"Wrote {path} ({path.stat().st_size:,} bytes): {sidecar['validation']}", flush=True)
    finally:
        implementation.apply_rotary_pos_emb = original_rotary
        implementation.repeat_kv = original_repeat
        for handle in handles:
            handle.remove()
    assert args.bundle.read_bytes() == main_bytes, "The main bundle must remain unchanged"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("static/traces/granite-moe.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("static/traces"))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--validate", action="store_true", help="Validate saved sidecars using standard-library Python")
    args = parser.parse_args()
    if args.validate:
        bundle = json.loads(args.bundle.read_text())
        for example in bundle["examples"]:
            path = args.output_dir / f"attention-operations-{example['id']}.json"
            print(example["id"], json.dumps(validate_sidecar(json.loads(path.read_text()), bundle)))
    else:
        capture(args)


if __name__ == "__main__":
    main()
