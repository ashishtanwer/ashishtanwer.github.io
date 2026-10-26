#!/usr/bin/env python3
"""Capture exact small expert-arithmetic views without changing the main bundle.

Uses the pinned environment in requirements-traces.txt and the model revision
already specified by granite-moe.json. All matrix tails are signed dot products
over the full omitted coordinates, independently computed in float64 from
observed activations and the actual pretrained weights.

    python scripts/capture_expert_operations.py --cache-dir /tmp/moe-hf-cache
    python scripts/capture_expert_operations.py --validate
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import math
from pathlib import Path
import platform

from capture_moe_traces import MODEL_ID, MODEL_REVISION, rounded, sha256_file


def silu(value):
    if value >= 0:
        return value / (1 + math.exp(-value))
    exp = math.exp(value)
    return value * exp / (1 + exp)


def validate_sidecar(sidecar, main):
    assert sidecar["schemaVersion"] == 1 and sidecar["kind"] == "expert-operations"
    assert sidecar["model"]["id"] == main["model"]["id"]
    assert sidecar["model"]["revision"] == main["model"]["revision"]
    example = next(example for example in main["examples"] if example["id"] == sidecar["example"]["id"])
    assert sidecar["example"]["prompt"] == example["prompt"]
    assert sidecar["example"]["tokens"] == example["tokens"]
    assert len(sidecar["layers"]) == main["model"]["numLayers"]
    expert_count = 0
    projection_count = 0
    maximum_projection_error = 0.0
    maximum_projection_ratio = 0.0
    maximum_sum_error = 0.0
    negative_tails = 0

    def vector(value, length=4):
        assert len(value) == length and all(math.isfinite(item) for item in value)

    def projection(record, inputs, input_dimensions):
        nonlocal projection_count, maximum_projection_error, maximum_projection_ratio, negative_tails
        projection_count += 1
        assert record["inputDimensions"] == input_dimensions
        assert record["omittedDimensions"] == input_dimensions - 4
        assert len(record["weights"]) == 4
        for row in record["weights"]:
            vector(row)
        for field in ("omittedContribution", "output", "reconstructionTolerance"):
            vector(record[field])
        for coordinate in range(4):
            tail = record["omittedContribution"][coordinate]
            negative_tails += int(tail < 0)
            reconstructed = sum(record["weights"][coordinate][i] * inputs[i] for i in range(4)) + tail
            error = abs(reconstructed - record["output"][coordinate])
            tolerance = record["reconstructionTolerance"][coordinate]
            assert tolerance > 0 and error <= tolerance
            maximum_projection_error = max(maximum_projection_error, error)
            maximum_projection_ratio = max(maximum_projection_ratio, error / tolerance)

    for layer_index, layer in enumerate(sidecar["layers"]):
        assert layer["index"] == layer_index
        assert len(layer["tokens"]) == len(example["tokens"])
        for token_index, token in enumerate(layer["tokens"]):
            original = example["layers"][layer_index]["tokens"][token_index]
            assert token["selectedExperts"] == original["selectedExperts"]
            assert token["routingWeights"] == original["routingWeights"]
            vector(token["inputPreview"])
            vector(token["combinedOutput"])
            assert token["combinedOutput"] == original["combinedOutputPreview"][:4]
            assert len(token["experts"]) == main["model"]["topK"]
            for rank, expert in enumerate(token["experts"]):
                expert_count += 1
                assert expert["id"] == token["selectedExperts"][rank]
                projection(expert["gate"], token["inputPreview"], main["model"]["hiddenSize"])
                projection(expert["up"], token["inputPreview"], main["model"]["hiddenSize"])
                projection(expert["down"], expert["product"], main["model"]["intermediateSize"])
                for field in ("siluGate", "product", "weightedOutput"):
                    vector(expert[field])
                assert expert["down"]["output"] == original["expertOutputPreview"][rank][:4]
                for coordinate in range(4):
                    assert math.isclose(expert["siluGate"][coordinate], silu(expert["gate"]["output"][coordinate]),
                                        rel_tol=2e-6, abs_tol=2e-6)
                    assert math.isclose(expert["product"][coordinate],
                                        expert["siluGate"][coordinate] * expert["up"]["output"][coordinate],
                                        rel_tol=2e-6, abs_tol=1e-5)
                    assert math.isclose(expert["weightedOutput"][coordinate],
                                        expert["down"]["output"][coordinate] * token["routingWeights"][rank],
                                        rel_tol=2e-6, abs_tol=1e-5)
            for coordinate in range(4):
                reconstructed = sum(expert["weightedOutput"][coordinate] for expert in token["experts"])
                actual = token["combinedOutput"][coordinate]
                maximum_sum_error = max(maximum_sum_error, abs(actual - reconstructed))
                assert math.isclose(actual, reconstructed, rel_tol=2e-6, abs_tol=1e-5)

    assert sidecar["capture"]["matchesMainVocabularyLogits"] is True
    return {
        "passed": True, "expertEvaluations": expert_count, "projections": projection_count,
        "signedNegativeOmittedTerms": negative_tails,
        "maxProjectionReconstructionError": maximum_projection_error,
        "maxProjectionToleranceRatio": maximum_projection_ratio,
        "maxWeightedSumPreviewError": maximum_sum_error,
    }


def capture(args):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    original_bytes = args.bundle.read_bytes()
    bundle_hash = sha256_file(args.bundle)
    main = json.loads(original_bytes)
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
    assert config.hidden_act == "silu"
    for filename in main["capture"]["modelFiles"]:
        assert sha256_file(snapshot / filename["name"]) == filename["sha256"]
    cache = {}
    layers = {}
    handles = []

    def router_hook(index):
        def hook(_module, inputs, output):
            sorted_indices, batch_indices, gates, expert_sizes, logits = output
            cache[index] = {"input": inputs[0].detach(), "sortedIndices": sorted_indices,
                            "batchIndices": batch_indices, "gates": gates, "logits": logits}
        return hook

    def input_hook(index):
        def hook(module, inputs, output):
            cache[index].update(linearInput=inputs[0].detach(), linearOutput=output.detach(),
                                inputWeights=module.weight.detach())
        return hook

    def activation_hook(index):
        def hook(_module, _inputs, output):
            cache[index]["silu"] = output.detach()
        return hook

    def output_hook(index):
        def hook(module, inputs, output):
            cache[index].update(product=inputs[0].detach(), downOutput=output.detach(),
                                outputWeights=module.weight.detach())
        return hook

    def project(weights, full_input, actual_output):
        # FP64 dot products over all omitted inputs, from the actual forward
        # pass. This signed tail is never obtained by subtracting the preview
        # from the output, and is never an activation or weight norm.
        rows = weights[:4].double()
        inputs = full_input.double()
        omitted = rows[:, 4:] @ inputs[4:]
        reconstructed = rows[:, :4] @ inputs[:4] + omitted
        actual = actual_output[:4].double()
        # Bound FP32 accumulation error by the sum of absolute dot-product
        # terms; this remains meaningful even when signed terms cancel.
        sum_absolute_terms = rows.abs() @ inputs.abs()
        tolerance = 1e-5 + 1e-6 * sum_absolute_terms
        error = (reconstructed - actual).abs()
        assert torch.isfinite(reconstructed).all() and torch.all(error <= tolerance), (error, tolerance)
        return {
            "weights": weights[:4, :4].tolist(),
            "omittedContribution": omitted.tolist(),
            "output": rounded(actual_output[:4].tolist()),
            "inputDimensions": full_input.numel(),
            "omittedDimensions": full_input.numel() - 4,
            "reconstructionTolerance": tolerance.tolist(),
            "reconstructionMaxAbsError": error.max().item(),
        }

    def moe_hook(index):
        def hook(_module, _inputs, output):
            data = cache.pop(index)
            actual_output, logits = output
            torch.testing.assert_close(data["logits"], logits, rtol=0, atol=0)
            selected_logits, selected_experts = logits.topk(config.num_experts_per_tok, dim=-1)
            selected_weights = selected_logits.softmax(dim=-1)
            inverse = data["sortedIndices"].argsort()
            gate, up = data["linearOutput"].chunk(2, dim=-1)
            torch.testing.assert_close(data["product"], data["silu"] * up, rtol=0, atol=0)
            torch.testing.assert_close(data["silu"], torch.nn.functional.silu(gate), rtol=0, atol=0)
            weighted_outputs = data["downOutput"] * data["gates"][:, None]
            reconstructed_full = torch.zeros_like(actual_output.reshape(-1, config.hidden_size)).index_add(
                0, data["batchIndices"], weighted_outputs)
            torch.testing.assert_close(reconstructed_full, actual_output[0], rtol=0, atol=0)
            token_rows = []
            for token_index in range(logits.shape[0]):
                expert_rows = []
                for rank, expert_id in enumerate(selected_experts[token_index].tolist()):
                    row = inverse[token_index * config.num_experts_per_tok + rank].item()
                    torch.testing.assert_close(data["linearInput"][row], data["input"][token_index], rtol=0, atol=0)
                    torch.testing.assert_close(data["gates"][row], selected_weights[token_index, rank], rtol=0, atol=0)
                    gate_weights, up_weights = data["inputWeights"][expert_id].chunk(2, dim=0)
                    expert_rows.append({
                        "id": expert_id,
                        "gate": project(gate_weights, data["linearInput"][row], gate[row]),
                        "up": project(up_weights, data["linearInput"][row], up[row]),
                        "siluGate": rounded(data["silu"][row, :4].tolist()),
                        "product": rounded(data["product"][row, :4].tolist()),
                        "down": project(data["outputWeights"][expert_id], data["product"][row], data["downOutput"][row]),
                        "weightedOutput": rounded(weighted_outputs[row, :4].tolist()),
                    })
                token_rows.append({
                    "inputPreview": rounded(data["input"][token_index, :4].tolist()),
                    "selectedExperts": selected_experts[token_index].tolist(),
                    "routingWeights": rounded(selected_weights[token_index].tolist()),
                    "experts": expert_rows,
                    "combinedOutput": rounded(actual_output[0, token_index, :4].tolist()),
                })
            layers[index] = {"index": index, "tokens": token_rows}
        return hook

    for index, layer in enumerate(model.model.layers):
        moe = layer.block_sparse_moe
        handles.extend([
            moe.router.register_forward_hook(router_hook(index)),
            moe.input_linear.register_forward_hook(input_hook(index)),
            moe.activation.register_forward_hook(activation_hook(index)),
            moe.output_linear.register_forward_hook(output_hook(index)),
            moe.register_forward_hook(moe_hook(index)),
        ])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for example in main["examples"]:
        print(f"Capturing expert operations: {example['id']}", flush=True)
        layers.clear()
        encoded = tokenizer(example["prompt"], return_tensors="pt", add_special_tokens=False)
        assert encoded["input_ids"][0].tolist() == [token["id"] for token in example["tokens"]]
        with torch.inference_mode():
            output = model(**encoded, use_cache=False, output_router_logits=True, return_dict=True)
        assert rounded(output.logits[0, -1].tolist()) == example["nextTokenLogits"]
        for index, logits in enumerate(output.router_logits):
            assert rounded(logits.tolist()) == [token["routerLogits"] for token in example["layers"][index]["tokens"]]
        sidecar = {
            "schemaVersion": 1, "kind": "expert-operations",
            "model": {key: main["model"][key] for key in
                      ("id", "revision", "hiddenSize", "intermediateSize", "topK", "numLayers", "numExperts")},
            "capture": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "script": "scripts/capture_expert_operations.py",
                "scriptSha256": sha256_file(Path(__file__)),
                "mainBundleSha256": bundle_hash,
                "mainCaptureScriptSha256": main["capture"]["scriptSha256"],
                "device": "cpu", "dtype": "float32", "omittedDotProductDtype": "float64",
                "python": platform.python_version(),
                "packages": {name: metadata.version(name) for name in
                             ("torch", "transformers", "safetensors", "huggingface-hub")},
                "matchesMainVocabularyLogits": True,
                "notes": [
                    "All selected experts, input tokens, and 24 layers are observed in an actual pretrained forward pass. The main bundle is never rewritten.",
                    "Input is the actual post-attention RMSNorm output entering the MoE. Gate and up projections are the two halves of the actual input_linear output.",
                    "siluGate is captured from the model's activation module; product is the actual full 512-coordinate input to output_linear. Product equals SiLU(gate) times up exactly in float32.",
                    "Each weight tile is the true first 4 output rows and first 4 input columns. No projection has a bias.",
                    "omittedContribution is the signed dot product over input coordinates 4 onward, independently computed in float64 using actual full activations and pretrained weights. It is not output minus visible terms, and is not a norm.",
                    "Down-projection omitted terms use all 508 remaining coordinates of the actual full SiLU(gate) times up product.",
                    "Projection tolerance is 1e-5 + 1e-6 times the sum of absolute full dot-product terms, allowing float32 accumulation error even under cancellation.",
                    "Actual projection/activation previews are rounded to 9 decimal places; weights and independently computed omitted terms retain their Python float precision.",
                    "Weighted outputs and combinedOutput are before the decoder's residual multiplication by 0.22. Full 1024-coordinate grouped weighted sums exactly match the actual model MoE output.",
                    "All recorded router logits and all 49152 final-token vocabulary logits exactly match the existing main bundle after its documented nine-decimal serialization.",
                ],
            },
            "example": {key: example[key] for key in ("id", "prompt", "tokens")},
            "layers": [layers[index] for index in range(config.num_hidden_layers)],
        }
        sidecar["validation"] = validate_sidecar(sidecar, main)
        path = args.output_dir / f"expert-operations-{example['id']}.json"
        path.write_text(json.dumps(sidecar, ensure_ascii=False, separators=(",", ":")) + "\n")
        assert path.stat().st_size < 25_000_000
        print(f"Wrote {path} ({path.stat().st_size:,} bytes): {sidecar['validation']}", flush=True)
    for handle in handles:
        handle.remove()
    assert args.bundle.read_bytes() == original_bytes, "The main bundle must remain untouched"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("static/traces/granite-moe.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("static/traces"))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--validate", action="store_true", help="Validate saved sidecars with standard-library Python")
    args = parser.parse_args()
    if args.validate:
        bundle = json.loads(args.bundle.read_text())
        for example in bundle["examples"]:
            path = args.output_dir / f"expert-operations-{example['id']}.json"
            print(example["id"], json.dumps(validate_sidecar(json.loads(path.read_text()), bundle)))
    else:
        capture(args)


if __name__ == "__main__":
    main()
