#!/usr/bin/env python3
"""Record genuine Granite MoE forward passes for the static explainer.

Install scripts/requirements-traces.txt in an isolated environment, then run:
    python scripts/capture_moe_traces.py --cache-dir /tmp/moe-hf-cache

The default model revision is immutable. Hooks observe the existing forward pass;
they neither replace the router nor alter activations. No model training occurs.
Validate a shipped bundle without installing ML packages:
    python scripts/capture_moe_traces.py --validate static/traces/granite-moe.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import sys
import time


MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-base"
MODEL_REVISION = "408b6e90baab8cf24f4aa9f8e19703ffa0a53b29"
EXAMPLES = [
    {"id": "language", "title": "A familiar fact", "category": "Language",
     "prompt": "The capital of France is"},
    {"id": "code", "title": "Complete a function", "category": "Code",
     "prompt": "def add(a, b):\n    return a +"},
    {"id": "math", "title": "A little arithmetic", "category": "Math",
     "prompt": "Question: What is 7 + 5?\nAnswer:"},
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def rounded(values, digits=9):
    """Compact JSON; preserve more precision than the UI displays."""
    if isinstance(values, list):
        return [rounded(value, digits) for value in values]
    return round(float(values), digits)


def validate_bundle(bundle: dict) -> dict:
    """Check actual recorded values, including the expert sum shown by the UI."""
    assert bundle["schemaVersion"] == 1
    config = bundle["model"]
    assert len(bundle["vocabulary"]) == config["vocabSize"]
    assert all(isinstance(text, str) for text in bundle["vocabulary"])
    n_experts, top_k = config["numExperts"], config["topK"]
    token_layer_count = 0
    attention_row_count = 0
    max_preview_error = 0.0
    max_recorded_error = 0.0
    max_vocabulary_probability_error = 0.0
    for example in bundle["examples"]:
        assert example["validation"]["hookedVsUnhookedMaxLogitError"] == 0
        n_tokens = len(example["tokens"])
        assert "".join(token["text"] for token in example["tokens"]) == example["prompt"]
        assert len(example["layers"]) == config["numLayers"]
        assert [token["index"] for token in example["tokens"]] == list(range(n_tokens))
        def validate_previews(previews, norms):
            assert len(previews) == len(norms) == n_tokens
            for vector, norm in zip(previews, norms):
                assert len(vector) == 8 and all(math.isfinite(value) for value in vector)
                assert math.isfinite(norm) and norm >= 0
                assert math.sqrt(sum(value * value for value in vector)) <= norm + 1e-5

        validate_previews(example["embeddingPreview"], example["embeddingNorms"])
        validate_previews(example["finalHiddenPreview"], example["finalHiddenNorms"])
        assert len(example["attention"]) == config["numLayers"]
        assert len(example["attentionScores"]) == len(example["attentionMaskedScores"]) == config["numLayers"]
        for layer_index, heads in enumerate(example["attention"]):
            assert len(heads) == config["attentionHeads"]
            score_heads = example["attentionScores"][layer_index]
            masked_heads = example["attentionMaskedScores"][layer_index]
            assert len(score_heads) == len(masked_heads) == len(heads)
            for head_index, matrix in enumerate(heads):
                assert len(matrix) == n_tokens
                assert len(score_heads[head_index]) == len(masked_heads[head_index]) == n_tokens
                for query, row in enumerate(matrix):
                    attention_row_count += 1
                    scores = score_heads[head_index][query]
                    masked = masked_heads[head_index][query]
                    assert len(scores) == len(masked) == n_tokens
                    assert all(math.isfinite(value) for value in scores + masked)
                    assert len(row) == n_tokens
                    assert all(math.isfinite(value) and 0 <= value <= 1 for value in row)
                    assert math.isclose(sum(row), 1, abs_tol=2e-7)
                    assert all(value == 0 for value in row[query + 1:]), "Future-token attention must be masked"
                    assert masked[:query + 1] == scores[:query + 1]
                    assert all(value == config["attentionMaskValue"] for value in masked[query + 1:])
                    exponentials = [math.exp(value - max(masked)) for value in masked]
                    denominator = sum(exponentials)
                    assert all(abs(value - exp / denominator) < 2e-7
                               for value, exp in zip(row, exponentials))
        for layer_index, layer in enumerate(example["layers"]):
            assert layer["index"] == layer_index
            assert len(layer["tokens"]) == n_tokens
            validate_previews(layer["attentionInputPreview"], layer["attentionInputNorms"])
            validate_previews(layer["attentionOutputPreview"], layer["attentionOutputNorms"])
            for kind in ("query", "key", "value"):
                assert len(layer[f"{kind}Preview"]) == len(layer[f"{kind}Norms"]) == config["attentionHeads"]
                for head_previews, head_norms in zip(layer[f"{kind}Preview"], layer[f"{kind}Norms"]):
                    validate_previews(head_previews, head_norms)
            counts = [0] * n_experts
            for token in layer["tokens"]:
                token_layer_count += 1
                probabilities = token["routerProbabilities"]
                logits = token["routerLogits"]
                selected = token["selectedExperts"]
                weights = token["routingWeights"]
                assert len(probabilities) == len(logits) == n_experts
                assert len(selected) == len(set(selected)) == len(weights) == top_k
                assert all(math.isfinite(x) for x in logits + probabilities + weights)
                assert all(0 <= p <= 1 for p in probabilities + weights)
                assert math.isclose(sum(probabilities), 1, abs_tol=2e-7)
                assert math.isclose(sum(weights), 1, abs_tol=2e-7)
                assert selected == sorted(range(n_experts), key=lambda i: logits[i], reverse=True)[:top_k]
                softmax_denominator = sum(math.exp(logits[i] - max(logits)) for i in range(n_experts))
                expected_probabilities = [math.exp(x - max(logits)) / softmax_denominator for x in logits]
                assert all(abs(a-b) < 2e-7 for a, b in zip(probabilities, expected_probabilities))
                selected_mass = sum(probabilities[i] for i in selected)
                assert all(abs(weight - probabilities[expert] / selected_mass) < 2e-7
                           for weight, expert in zip(weights, selected))
                expert_preview = token["expertOutputPreview"]
                combined_preview = token["combinedOutputPreview"]
                assert len(expert_preview) == top_k
                assert len(combined_preview) == 8
                assert all(len(row) == 8 for row in expert_preview)
                for coordinate in range(8):
                    reconstructed = sum(weights[i] * expert_preview[i][coordinate] for i in range(top_k))
                    error = abs(reconstructed - combined_preview[coordinate])
                    max_preview_error = max(max_preview_error, error)
                    assert math.isclose(reconstructed, combined_preview[coordinate], rel_tol=2e-6, abs_tol=1e-5)
                assert len(token["expertOutputNorms"]) == len(token["weightedOutputNorms"]) == top_k
                assert all(math.isclose(out * weight, weighted, rel_tol=2e-6, abs_tol=1e-6)
                           for out, weight, weighted in zip(token["expertOutputNorms"], weights,
                                                            token["weightedOutputNorms"]))
                error = token["reconstructionMaxAbsError"]
                assert math.isfinite(error) and error >= 0
                assert token["reconstructionToleranceRatio"] <= 1
                assert error <= 1e-5 + 2e-6 * token["outputMaxAbs"]
                max_recorded_error = max(max_recorded_error, error)
                for expert in selected:
                    counts[expert] += 1
            assert layer["expertTokenCounts"] == counts
            assert sum(counts) == n_tokens * top_k
        candidates = example["nextTokenCandidates"]
        assert candidates and all(0 <= candidate["probability"] <= 1 for candidate in candidates)
        assert all(candidates[i]["probability"] >= candidates[i+1]["probability"]
                   for i in range(len(candidates)-1))
        assert sum(candidate["probability"] for candidate in candidates) <= 1 + 1e-7
        logits = example["nextTokenLogits"]
        assert len(logits) == config["vocabSize"] and all(math.isfinite(value) for value in logits)
        max_logit = max(logits)
        denominator = sum(math.exp(value - max_logit) for value in logits)
        for candidate in candidates:
            assert candidate["text"] == bundle["vocabulary"][candidate["id"]]
            error = abs(candidate["probability"] - math.exp(logits[candidate["id"]] - max_logit) / denominator)
            max_vocabulary_probability_error = max(max_vocabulary_probability_error, error)
            # Full-vocabulary float32 reduction can differ from Python float64.
            assert error < 5e-5, f"Vocabulary probability error: {error}"
    return {"passed": True, "examples": len(bundle["examples"]),
            "tokenLayerPairs": token_layer_count,
            "attentionRows": attention_row_count,
            "maxSerializedPreviewReconstructionError": max_preview_error,
            "maxFullVectorReconstructionError": max_recorded_error,
            "maxVocabularySoftmaxFloat32VsFloat64Error": max_vocabulary_probability_error,
            "checks": ["all input tokens and all routed layers present", "router softmax matches logits",
                       "selected experts match top-k logits", "selected weights sum to one",
                       "weighted expert previews reconstruct captured MoE output",
                       "full-vector reconstruction within atol=1e-5 plus rtol=2e-6", "expert counts match routing",
                       "hooked and uninstrumented passes have identical vocabulary logits",
                       "all attention heads present; rows sum to one and future positions are exactly zero",
                       "true scaled QK scores plus actual causal mask reproduce captured attention softmax",
                       "full vocabulary logits reproduce next-token candidates without top-10 renormalization",
                       "embedding, final hidden, and attention vectors have finite previews and full-vector norms",
                       "next-token candidates are sorted vocabulary probabilities"]}


def capture(args):
    # Keep validation available with only Python's standard library.
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.granitemoe.modeling_granitemoe import apply_rotary_pos_emb, repeat_kv

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    snapshot = Path(snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        allow_patterns=["*.json", "*.safetensors", "README.md", "LICENSE", "merges.txt", "vocab.json"],
    ))
    print(f"Loading {MODEL_ID}@{MODEL_REVISION} on {args.device} in float32", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, attn_implementation="eager",
    ).to(args.device).eval()
    config = model.config
    assert config.model_type == "granitemoe", "This hook layout is specific to Granite MoE."
    top_k = config.num_experts_per_tok
    n_experts = config.num_local_experts
    n_layers = config.num_hidden_layers
    current_layers = {}
    router_cache = {}
    expert_cache = {}
    attention_cache = {}
    projection_cache = {}
    attention_score_cache = {}
    handles = []

    def projection_hook(index, kind):
        def hook(_module, _inputs, output):
            projection_cache.setdefault(index, {})[kind] = output.detach()
        return hook

    def attention_hook(index):
        def hook(module, _inputs, kwargs, output):
            # Inputs are after input_layernorm. Outputs are after o_proj, before
            # the decoder scales the branch and adds the residual.
            attention_input = kwargs["hidden_states"].detach().cpu()[0]
            attention_output = output[0].detach().cpu()[0]
            projections = projection_cache.pop(index)
            batch_size, length, _ = kwargs["hidden_states"].shape
            query = projections["query"].view(batch_size, length, module.num_heads, module.head_dim).transpose(1, 2)
            key = projections["key"].view(batch_size, length, module.num_key_value_heads, module.head_dim).transpose(1, 2)
            value = projections["value"].view(batch_size, length, module.num_key_value_heads, module.head_dim).transpose(1, 2)
            cos, sin = kwargs["position_embeddings"]
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
            # Recompute the exact attention arithmetic from observed projection
            # outputs and the actual RoPE/mask inputs. Never infer QK from log(p).
            scores = torch.matmul(query, key.transpose(2, 3)) * module.scaling
            masked_scores = scores + kwargs["attention_mask"][:, :, :, :key.shape[-2]]
            reconstructed_attention = masked_scores.softmax(dim=-1, dtype=torch.float32).to(query.dtype)
            torch.testing.assert_close(reconstructed_attention, output[1], rtol=0, atol=0)
            attention_score_cache[index] = {
                "scores": rounded(scores.detach().cpu()[0].tolist()),
                "masked": rounded(masked_scores.detach().cpu()[0].tolist()),
            }
            attention_cache[index] = {
                "attentionInputPreview": rounded(attention_input[:, :8].tolist()),
                "attentionInputNorms": rounded(attention_input.norm(dim=-1).tolist()),
                "attentionOutputPreview": rounded(attention_output[:, :8].tolist()),
                "attentionOutputNorms": rounded(attention_output.norm(dim=-1).tolist()),
            }
            for kind, tensor in (("query", query), ("key", key), ("value", value)):
                head_vectors = tensor.detach().cpu()[0]
                attention_cache[index][f"{kind}Preview"] = rounded(head_vectors[:, :, :8].tolist())
                attention_cache[index][f"{kind}Norms"] = rounded(head_vectors.norm(dim=-1).tolist())
        return hook

    def router_hook(index):
        def hook(_module, inputs, output):
            sorted_indices, batch_indices, batch_gates, expert_sizes, logits = output
            router_cache[index] = {
                "sorted_indices": sorted_indices.detach().cpu(),
                "batch_indices": batch_indices.detach().cpu(),
                "batch_gates": batch_gates.detach().cpu(),
                "expert_sizes": expert_sizes,
                "logits": logits.detach().cpu(),
                "input": inputs[0].detach().cpu(),
            }
        return hook

    def expert_hook(index):
        def hook(_module, _inputs, output):
            # Actual expert outputs, before the model multiplies by routing gates.
            expert_cache[index] = output.detach().cpu()
        return hook

    def moe_hook(index):
        def hook(_module, _inputs, output):
            state = router_cache.pop(index)
            grouped_outputs = expert_cache.pop(index)
            actual_outputs, actual_logits = output
            actual_outputs = actual_outputs.detach().cpu().reshape(-1, config.hidden_size)
            torch.testing.assert_close(actual_logits.detach().cpu(), state["logits"], rtol=0, atol=0)
            logits = state["logits"]
            token_count = logits.shape[0]
            selected_logits, selected_experts = logits.topk(top_k, dim=-1)
            expected_weights = selected_logits.softmax(dim=-1)
            # Undo the model's grouping by expert, preserving its exact token/top-k ordering.
            flat_outputs = torch.empty_like(grouped_outputs)
            flat_outputs[state["sorted_indices"]] = grouped_outputs
            expert_outputs = flat_outputs.reshape(token_count, top_k, config.hidden_size)
            flat_weights = torch.empty_like(state["batch_gates"])
            flat_weights[state["sorted_indices"]] = state["batch_gates"]
            weights = flat_weights.reshape(token_count, top_k)
            torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
            weighted_outputs = expert_outputs * weights[:, :, None]
            reconstructed = weighted_outputs.sum(dim=1)
            errors = (actual_outputs - reconstructed).abs().amax(dim=-1)
            tolerance_ratios = ((actual_outputs - reconstructed).abs() /
                                (1e-5 + 2e-6 * actual_outputs.abs())).amax(dim=-1)
            torch.testing.assert_close(reconstructed, actual_outputs, rtol=2e-6, atol=1e-5)
            probabilities = logits.softmax(dim=-1)
            expert_norms = expert_outputs.norm(dim=-1)
            weighted_norms = weighted_outputs.norm(dim=-1)
            tokens = []
            for token_index in range(token_count):
                tokens.append({
                    "routerLogits": rounded(logits[token_index].tolist()),
                    "routerProbabilities": rounded(probabilities[token_index].tolist()),
                    "selectedExperts": selected_experts[token_index].tolist(),
                    "routingWeights": rounded(weights[token_index].tolist()),
                    "expertOutputNorms": rounded(expert_norms[token_index].tolist()),
                    "weightedOutputNorms": rounded(weighted_norms[token_index].tolist()),
                    "expertOutputPreview": rounded(expert_outputs[token_index, :, :8].tolist()),
                    "combinedOutputPreview": rounded(actual_outputs[token_index, :8].tolist()),
                    "inputNorm": round(state["input"][token_index].norm().item(), 9),
                    "outputNorm": round(actual_outputs[token_index].norm().item(), 9),
                    "outputMaxAbs": actual_outputs[token_index].abs().max().item(),
                    "reconstructionMaxAbsError": errors[token_index].item(),
                    "reconstructionToleranceRatio": tolerance_ratios[token_index].item(),
                })
            current_layers[index] = {"index": index, "tokens": tokens,
                                     "expertTokenCounts": state["expert_sizes"],
                                     **attention_cache.pop(index)}
        return hook

    for index, layer in enumerate(model.model.layers):
        moe = layer.block_sparse_moe
        handles.extend([
            layer.self_attn.q_proj.register_forward_hook(projection_hook(index, "query")),
            layer.self_attn.k_proj.register_forward_hook(projection_hook(index, "key")),
            layer.self_attn.v_proj.register_forward_hook(projection_hook(index, "value")),
            layer.self_attn.register_forward_hook(attention_hook(index), with_kwargs=True),
            moe.router.register_forward_hook(router_hook(index)),
            moe.output_linear.register_forward_hook(expert_hook(index)),
            moe.register_forward_hook(moe_hook(index)),
        ])

    recorded_examples = []
    reference_logits = []
    for example in EXAMPLES:
        start = time.monotonic()
        print(f"Capturing {example['id']}: {example['prompt']!r}", flush=True)
        current_layers.clear()
        attention_score_cache.clear()
        encoded = tokenizer(example["prompt"], return_tensors="pt", add_special_tokens=False).to(args.device)
        input_ids = encoded["input_ids"][0].tolist()
        with torch.inference_mode():
            output = model(**encoded, use_cache=False, output_router_logits=True,
                           output_attentions=True, output_hidden_states=True, return_dict=True)
        assert len(current_layers) == len(output.router_logits) == n_layers
        assert len(output.attentions) == n_layers
        assert len(output.hidden_states) == n_layers + 1
        embedding = output.hidden_states[0].detach().cpu()[0]
        final_hidden = output.hidden_states[-1].detach().cpu()[0]
        # Verify the public hidden-state output means scaled token embeddings,
        # rather than silently labeling a normalized or position-added vector.
        expected_embedding = (model.model.embed_tokens(encoded["input_ids"]) * config.embedding_multiplier)
        torch.testing.assert_close(embedding, expected_embedding.detach().cpu()[0], rtol=0, atol=0)
        for index, router_logits in enumerate(output.router_logits):
            captured_logits = torch.tensor([token["routerLogits"] for token in current_layers[index]["tokens"]])
            torch.testing.assert_close(captured_logits, router_logits.detach().cpu(), rtol=1e-6, atol=1e-7)
        probabilities = output.logits[0, -1].float().softmax(dim=-1)
        probabilities_float64 = output.logits[0, -1].double().softmax(dim=-1)
        reference_logits.append(output.logits.detach().cpu().clone())
        candidate_probabilities, candidate_ids = probabilities.topk(10)
        elapsed = time.monotonic() - start
        record = {
            **example,
            "tokens": [{"index": index, "id": token_id,
                        "text": tokenizer.decode([token_id], clean_up_tokenization_spaces=False)}
                       for index, token_id in enumerate(input_ids)],
            "layers": [current_layers[index] for index in range(n_layers)],
            "attention": [rounded(attention.detach().cpu()[0].tolist()) for attention in output.attentions],
            "attentionScores": [attention_score_cache[index]["scores"] for index in range(n_layers)],
            "attentionMaskedScores": [attention_score_cache[index]["masked"] for index in range(n_layers)],
            "embeddingPreview": rounded(embedding[:, :8].tolist()),
            "embeddingNorms": rounded(embedding.norm(dim=-1).tolist()),
            "finalHiddenPreview": rounded(final_hidden[:, :8].tolist()),
            "finalHiddenNorms": rounded(final_hidden.norm(dim=-1).tolist()),
            "nextTokenCandidates": [
                {"id": token_id, "text": tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
                 "probability": round(probability, 9)}
                for token_id, probability in zip(candidate_ids.tolist(), candidate_probabilities.tolist())],
            "nextTokenLogits": rounded(output.logits[0, -1].float().detach().cpu().tolist()),
            "validation": {"capturedTokenCount": len(input_ids), "capturedLayerCount": n_layers,
                           "forwardSeconds": round(elapsed, 3),
                           "vocabularyFloat32ProbabilitySum": probabilities.double().sum().item(),
                           "vocabularyFloat32VsFloat64MaxDifference": (
                               probabilities.double() - probabilities_float64).abs().max().item(),
                           "maxFullVectorReconstructionError": max(
                               token["reconstructionMaxAbsError"]
                               for layer in current_layers.values() for token in layer["tokens"])},
        }
        recorded_examples.append(record)
        print(f"  {len(input_ids)} tokens × {n_layers} layers in {elapsed:.2f}s", flush=True)

    for handle in handles:
        handle.remove()
    print("Verifying hooks preserved the full vocabulary logits against uninstrumented passes", flush=True)
    for example, captured_logits in zip(recorded_examples, reference_logits):
        encoded = tokenizer(example["prompt"], return_tensors="pt", add_special_tokens=False).to(args.device)
        with torch.inference_mode():
            plain_output = model(**encoded, use_cache=False, return_dict=True)
        plain_logits = plain_output.logits.detach().cpu()
        torch.testing.assert_close(captured_logits, plain_logits, rtol=0, atol=0)
        example["validation"]["hookedVsUnhookedMaxLogitError"] = (
            captured_logits - plain_logits).abs().max().item()
    print("Hashing downloaded model weight shards for provenance", flush=True)
    packages = {distribution.metadata["Name"]: distribution.version
                for distribution in metadata.distributions()}
    bundle = {
        "schemaVersion": 1,
        "model": {
            "id": MODEL_ID, "name": "Granite 3.1 1B-A400M", "revision": MODEL_REVISION,
            "url": f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}",
            "license": "Apache-2.0", "numLayers": n_layers, "numExperts": n_experts,
            "topK": top_k, "hiddenSize": config.hidden_size,
            "intermediateSize": config.intermediate_size,
            "totalParameters": sum(parameter.numel() for parameter in model.parameters()),
            "activeParametersApprox": 400_000_000,
            "residualMultiplier": config.residual_multiplier,
            "attentionHeads": config.num_attention_heads,
            "keyValueHeads": config.num_key_value_heads,
            "attentionHeadSize": config.hidden_size // config.num_attention_heads,
            "attentionMultiplier": config.attention_multiplier,
            "embeddingMultiplier": config.embedding_multiplier,
            "logitsScaling": config.logits_scaling,
            "ropeTheta": config.rope_theta,
            "vocabSize": config.vocab_size,
            "attentionMaskValue": torch.finfo(torch.float32).min,
        },
        "capture": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": args.device, "dtype": "float32", "python": platform.python_version(),
            "platform": platform.platform(), "packages": dict(sorted(packages.items())),
            "script": "scripts/capture_moe_traces.py", "scriptSha256": sha256_file(Path(__file__)),
            "modelFiles": [{"name": path.name, "bytes": path.stat().st_size,
                            "sha256": sha256_file(path)}
                           for path in sorted(snapshot.glob("*.safetensors"))],
            "configSha256": sha256_file(snapshot / "config.json"),
            "attentionImplementation": "eager", "seed": 0, "threads": args.threads,
            "notes": [
                "Actual pretrained model execution in eval mode; no simulated routing or random model weights.",
                "Each example is one full forward pass over the supplied prompt, with no generated continuation.",
                "Forward hooks observe actual router logits, dispatch ordering, normalized gates, expert outputs, and MoE outputs.",
                "Attention matrices are the actual post-softmax weights returned by eager attention, shaped [layer][head][query][key]. Eval mode disables attention dropout; future-token entries are exactly zero.",
                "attentionScores and attentionMaskedScores are recomputed from actual observed Q/K projection outputs and the forward pass's actual RoPE and causal mask, using the same operations as Granite's eager attention. Their softmax must exactly match the model's returned attention. Scores include the configured 1/64 scaling; masked future entries are the actual float32 minimum value.",
                "queryPreview/keyPreview/valuePreview contain the first 8 of 64 coordinates for each head and token, with full 64-coordinate norms. Queries and keys are after RoPE; keys and values repeat each of the 8 KV heads across its 2 query heads, matching grouped-query attention.",
                "embeddingPreview is the token embedding lookup multiplied by 12, before the first layer. Granite uses rotary positional embeddings in attention; no learned positional vector is added to these embeddings.",
                "attentionInputPreview is after each layer's input RMSNorm. attentionOutputPreview is after the attention output projection, before multiplication by 0.22 and residual addition.",
                "finalHiddenPreview is after the final RMSNorm, before the LM head; the model divides LM-head logits by 6.",
                "Every example is rerun without hooks; all vocabulary logits at every input position must match exactly.",
                "Granite selects top-8 logits, then softmaxes those 8. routerProbabilities softmaxes all 32 captured logits for comparison; routingWeights are the actual gates used by the model.",
                "Expert IDs and layer indices are zero-based. Selected arrays are ordered by descending router logit.",
                "Previews show the first 8 of 1024 coordinates; norms and reconstruction validation use all 1024 coordinates.",
                "Combined MoE outputs precede the decoder's residual multiplication by 0.22 and residual addition.",
                "Next-token probabilities are from the full 49,152-token vocabulary after final normalization and LM head, without sampling temperature.",
                "nextTokenLogits stores all final-token model logits, already divided by Granite's inherent logitsScaling=6. vocabulary maps every vocabulary ID to tokenizer.decode([ID]) with whitespace preserved. These allow exact full-vocabulary temperature and sampling demonstrations without rerunning the model.",
                "The recorded nextTokenCandidates preserve PyTorch CPU float32 softmax outputs. A 49,152-way float32 reduction differs slightly from stable float64 softmax; each example records its full float32 probability mass and maximum float32-vs-float64 difference. Browser float64 recomputation uses the same real logits and may differ within this measured precision effect.",
                "No semantic expert specialties are inferred. Experts with the same ID in different layers have different parameters.",
                "Serialized numeric arrays are rounded to 9 decimal places. Full-vector reconstruction errors are measured before rounding.",
                "Float32 reduction order can change last bits. All 1024 reconstructed coordinates pass PyTorch assert_close with atol=1e-5 and rtol=2e-6; reconstructionToleranceRatio is the maximum coordinate error divided by its allowed tolerance.",
            ],
        },
        "examples": recorded_examples,
        "vocabulary": [tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                       for token_id in range(config.vocab_size)],
    }
    bundle["validation"] = validate_bundle(bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps(bundle["validation"], indent=2))
    print(f"Wrote {args.output} ({args.output.stat().st_size:,} bytes)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("static/traces/granite-moe.json"))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--validate", type=Path, help="Validate an existing bundle; no ML dependencies required")
    args = parser.parse_args()
    if args.validate:
        result = validate_bundle(json.loads(args.validate.read_text()))
        print(json.dumps(result, indent=2))
    else:
        capture(args)


if __name__ == "__main__":
    main()
