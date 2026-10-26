# Recorded Granite MoE traces

`granite-moe.json` contains actual forward-pass measurements from the pretrained
[IBM Granite 3.1 1B-A400M base model](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-base/tree/408b6e90baab8cf24f4aa9f8e19703ffa0a53b29),
revision `408b6e90baab8cf24f4aa9f8e19703ffa0a53b29`. The model weights are
Apache-2.0 licensed. The expert-operation sidecars include small 4×4 excerpts of
those pretrained weights; the full model weights are not distributed here.

Captured on October 15, 2026 at 23:58:46 UTC using CPU, float32, Python 3.12.12,
PyTorch 2.6.0, and Transformers 4.51.3. The bundle includes the full package
versions, model-shard SHA-256 hashes, capture-script hash, and validation results.

The three prompts contain 29 tokens total. Every token is recorded at all 24
MoE layers: 696 token/layer pairs and 5,568 selected expert evaluations. Each
layer contains 32 independently parameterized experts and routes each token to 8.
The same forward passes also record all 16 attention heads at each layer:
1,152 complete attention matrices and 11,136 query rows across the three prompts.
The bundle also includes raw and masked attention scores, real Q/K/V previews,
all 49,152 final-token logits per example, and one shared decoded vocabulary.
Its JSON size is about 12.6 MB, or 4.5 MB when gzip-compressed.

| Example  | Exact prompt                                                   | Tokens |
| -------- | -------------------------------------------------------------- | -----: |
| Language | `The capital of France is`                                     |      5 |
| Code     | `def add(a, b):` followed by a newline and `    return a +`    |     11 |
| Math     | `Question: What is 7 + 5?` followed by a newline and `Answer:` |     13 |

## What the numbers mean

- `attention`: actual post-softmax attention weights with shape
  `[layer][head][query token][key token]`. All rows sum to one and all positions
  after the query token are exactly zero. Attention dropout is disabled in eval
  mode. Granite uses 16 query heads, 8 key/value heads, head dimension 64, RoPE,
  and its configured attention-logit multiplier of 1/64.
- `attentionScores`: scaled QK scores before masking, reconstructed using the
  actual observed Q/K projection outputs and that forward pass's RoPE inputs.
  `attentionMaskedScores` adds the actual causal mask. Their softmax is checked
  to match the model-returned attention exactly. Masked future entries contain
  the actual float32 minimum, about -3.4028e38; the UI can display that sentinel
  as negative infinity. These scores are never inferred from log probabilities.
- Per-layer `queryPreview`, `keyPreview`, and `valuePreview`: shape
  `[head][token][8 coordinates]`, with corresponding `queryNorms`, `keyNorms`,
  and `valueNorms` computed over all 64 head coordinates. Queries and keys are
  after RoPE. The 8 key/value heads are each repeated for their 2 query heads,
  matching the model's grouped-query attention.
- `embeddingPreview` / `embeddingNorms`: the first eight coordinates and full
  L2 norm of each token's embedding lookup multiplied by 12, before layer 1.
  No learned positional vector is added; Granite applies rotary positions to
  queries and keys inside attention.
- Per-layer `attentionInputPreview` / `attentionInputNorms`: the attention
  module's actual input after the layer's input RMSNorm.
- Per-layer `attentionOutputPreview` / `attentionOutputNorms`: the attention
  module's actual output after its output projection, before multiplication by
  0.22 and addition to the residual stream.
- `finalHiddenPreview` / `finalHiddenNorms`: the last hidden vectors after the
  model's final RMSNorm and before its language-model head. The head's resulting
  logits are divided by the model's configured factor of 6 before softmax.
- `routerLogits`: captured from the actual router. Granite selects the 8 largest
  logits and applies softmax to those 8 to obtain its actual `routingWeights`.
- `routerProbabilities`: softmax over all 32 captured logits, computed for the
  visualization. These full-distribution probabilities are distinct from the
  selected routing weights used to combine outputs.
- `selectedExperts`: zero-based expert IDs, sorted by router score. All selected
  output and weight arrays use this order. An expert ID refers only to its layer;
  expert 0 in layer 1 has different parameters from expert 0 in layer 2.
- `expertOutputPreview` and `combinedOutputPreview`: the first 8 of 1,024 actual
  coordinates. The complete 1,024-coordinate vectors are used for norms and
  reconstruction verification, then discarded to keep the website small.
- `combinedOutputPreview`: the weighted MoE sum before multiplication by the
  decoder's residual multiplier (0.22) and addition to the residual stream.
- `nextTokenCandidates`: the 10 highest probabilities from the complete
  49,152-token vocabulary. They are not renormalized to sum to one. A token may
  be whitespace or part of a word: the language example predicts ` Par`, while
  the math example's most likely next token is a space.
- `nextTokenLogits`: all 49,152 final-token model logits, including the model's
  inherent division by 6. Shared `vocabulary` maps every vocabulary ID to its
  decoded text, preserving whitespace. Together these allow temperature,
  top-k, and top-p demonstrations over the full saved vocabulary distribution.
  They do not provide inference for new prompts or subsequent generated tokens.

The data does not assign semantic specialties to experts, simulate routing,
change expert-routing top-k, or generate a continuation. Choosing a token or layer explores
the recorded forward pass.

## Reproduce the capture

From the repository root, create an isolated Python environment and install:

```sh
python3 -m venv .trace-venv
.trace-venv/bin/python -m pip install -r scripts/requirements-traces.txt
TOKENIZERS_PARALLELISM=false .trace-venv/bin/python scripts/capture_moe_traces.py \
  --cache-dir .hf-cache
```

The first run downloads about 2.67 GB of public model weights. Inference uses
float32 on CPU by default. The output is `static/traces/granite-moe.json`.
The exact script and requirements are also served in this directory for download.

Validate the existing artifact using only the Python standard library:

```sh
python3 scripts/capture_moe_traces.py --validate static/traces/granite-moe.json
```

The repository additionally validates it with:

```sh
node scripts/validate-traces.mjs
npm test
```

Forward hooks observe model outputs without modifying them. Each prompt is
rerun without hooks, and all vocabulary logits at every input position must
match exactly; the recorded maximum differences are all zero. Reconstructed
expert sums pass `torch.testing.assert_close` on every coordinate using
`atol=1e-5` and `rtol=2e-6`. Float32 summation order produces a largest absolute
difference of 0.0009765625 on a large-magnitude coordinate; the largest
error/tolerance ratio is 0.100616, below the required limit of 1. The maximum
reconstruction error in the eight displayed coordinates is 0.000002345.

The attention and full-vocabulary extensions were captured from the same pinned model and prompts.
All original MoE measurements and next-token probabilities are exactly unchanged
from the earlier MoE-only capture. The embedding interpretation is independently
checked against `embed_tokens(input_ids) * embedding_multiplier` during capture.

Numeric arrays are rounded to nine decimal places when serialized; full-vector
checks run before rounding. Reproducing on another device or software version
can change floating-point results slightly. The pinned source revision,
original capture environment, and artifact validation make those differences
inspectable.

The original candidate probabilities preserve PyTorch CPU float32 softmax.
Independent uninstrumented passes confirmed that the large vocabulary reduction
differs from stable float64 softmax by at most 0.000028343 in probability. The
bundle records the measured difference and full float32 probability sum for
each example. Browser sampling recomputes from the real full logits in float64,
so its unsampled probabilities can differ by this small precision effect.

## Inspecting multiplication inside an expert

The separately loaded `expert-operations-language.json`,
`expert-operations-code.json`, and `expert-operations-math.json` files record
all 5,568 selected expert evaluations. They are approximately 2.19 MB, 4.81 MB,
and 5.69 MB respectively. The main `granite-moe.json` bundle is unchanged.

Each expert shows its actual SwiGLU computation:

1. The post-attention RMSNorm input enters independent gate and up projections.
2. The model applies SiLU to the gate output.
3. It multiplies SiLU(gate) and up element by element.
4. A down projection maps the resulting 512-coordinate vector to 1,024 coordinates.
5. The router weight multiplies the expert output, and the selected experts' vectors
   are summed. This sum is before residual multiplication by 0.22.

For every gate, up, and down projection, the sidecar records a real 4×4 weight
tile, the first four actual output coordinates, and a signed
`omittedContribution` for each output coordinate. This term is independently
computed from **all** omitted coordinates using the actual full input and
pretrained weights in float64. Gate/up terms cover the remaining 1,020 input
coordinates; down terms cover the remaining 508 coordinates of the actual full
SiLU(gate) × up vector. The omitted term is never a norm or a residual invented
by subtracting the visible dot product from its answer.

Matrix rows are output coordinates and columns are input coordinates. Thus,
for each displayed output, the four visible input/weight products plus its
signed omitted-coordinate term reconstruct the actual projection output.
Float32 accumulation is checked with the scale-aware bound
`1e-5 + 1e-6 * sum(abs(all dot-product terms))`, including cancellation cases.
Every one of the 16,704 captured projections passes; the largest error/tolerance
ratio is below 0.381. The capture also verifies the actual SiLU module output,
the full elementwise product, and exact full-vector weighted sums.

Each sidecar includes independent provenance, its capture-script SHA-256, and
the SHA-256 of the unchanged main bundle. All router logits and all final-token
vocabulary logits are verified against that main recording. The downloadable
`capture_expert_operations.py` requires the adjacent `capture_moe_traces.py`
helper and the same pinned Python dependencies.

Reproduce or validate from the repository root:

```sh
TOKENIZERS_PARALLELISM=false .trace-venv/bin/python scripts/capture_expert_operations.py \
  --cache-dir .hf-cache
python3 scripts/capture_expert_operations.py --validate
node scripts/validate-expert-operations.mjs
```

## Inspecting head splitting, rotary positions, and shared K/V

The lazily loaded `attention-operations-language.json`,
`attention-operations-code.json`, and `attention-operations-math.json` files
contain complete 64-coordinate vectors, approximately 5.37 MB, 11.82 MB, and
13.97 MB respectively. The main bundle and expert-operation sidecars remain
unchanged.

The capture observes the actual pretrained projection outputs and the original
installed Granite implementation's `apply_rotary_pos_emb` and `repeat_kv`
executions. Transparent wrappers call those original functions, record their
actual inputs and outputs, and return their original tensors unchanged. The
post-RoPE vectors are captured from that execution, rather than produced solely
by an independent reconstruction.

- `queryBefore` and `queryAfter` have shape `[16 heads][tokens][64 coordinates]`.
  The pre-rotation vectors are verified to be the exact reshape of the actual
  1,024-coordinate query projection output.
- `keyBefore`, `keyAfter`, and `value` retain the 8 unique K/V heads, with shape
  `[8 heads][tokens][64 coordinates]`. Their pre-rotation vectors exactly match
  the actual 512-coordinate projection outputs.
- `positions` records the actual zero-based token positions. `rotary.cos` and
  `rotary.sin` contain 32 actual coefficients per position. The implementation's
  repeated second halves are verified identical, as are the coefficients used
  by every layer.
- RoPE pairs coordinate `i` with `i+32`, for `i=0..31`. For pair `(a,b)`, the
  actual outputs satisfy `a' = a*cos - b*sin` and `b' = b*cos + a*sin`. Position
  zero leaves all coordinates unchanged.
- Query head `h` uses unique key/value head `floor(h/2)`. The actual outputs of
  `repeat_kv` are verified to follow this mapping exactly. V is checked directly
  against the actual input to its repetition operation, proving it is unchanged
  by rotary position encoding.

Validation reconstructs all 534,528 recorded coordinate rotations and all
120,960 scaled QK scores, and compares every post-rotation/QKV preview to the
main recording. Original forward-pass attention probabilities, router logits,
and full vocabulary logits remain identical. Float64 reconstructions allow
small float32 accumulation differences using scale-aware tolerances; the actual
captured values are retained.

Each file records its own capture metadata, source SHA-256, and the unchanged
main-bundle SHA-256. The downloadable `capture_attention_operations.py` uses
the adjacent `capture_moe_traces.py` helper and the same pinned dependencies.

```sh
TOKENIZERS_PARALLELISM=false .trace-venv/bin/python scripts/capture_attention_operations.py \
  --cache-dir .hf-cache
python3 scripts/capture_attention_operations.py --validate
```

## Separate 128-coordinate RoPE example

`rope-example-128.json` is a separate, lazily loaded 49.9 KB recording from
[Qwen3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca),
pinned to revision `c1899de289a04d12100db370d81485cdf75e47ca`. It records block 0,
query head 0 and unique key/value head 0 for the five tokens in
`The capital of France is`. The Granite model diagram, main recording and all
existing sidecars retain their original values and SHA-256 hashes.

This model has 128 coordinates per head, 16 query heads, 8 key/value heads,
28 blocks, and `rope_theta=1000000` with no RoPE scaling. Its attention scores
use `1/sqrt(128)`. Qwen applies per-head `q_norm` and `k_norm` before RoPE;
the stored `queryBefore` and `keyBefore` values are the actual normalized
inputs at that boundary. All five Q/K/V arrays have shape `[tokens][128]`.
The stored actual `rotary.cos` and `rotary.sin` arrays have shape `[tokens][64]`,
with the implementation's repeated second halves verified identical. RoPE
pairs coordinate `i` with `i+64` for all `i=0..63`, so coordinates 0 through 127
can be inspected.

The capture wraps the original Transformers `apply_rotary_pos_emb` execution
and records its actual outputs, returning the original tensors to the model.
It verifies normalized Q/K outputs exactly match the observed RoPE inputs,
the V projection reshaped into heads exactly matches the unchanged V received
by attention, and actual grouped K/V repetition is exact. An unhooked second
execution matches every vocabulary logit at every prompt position exactly.
The independent validator reconstructs all 640 stored Q/K pairs; the largest
absolute rotation error is below `8.46e-6`, within the documented float32
tolerance. Actual position coefficients also match the configured theta.

The artifact includes capture time, package versions, CPU/float32 precision,
the immutable revision, verified weight and tokenizer checksums, configuration
checksum, installed model-implementation checksum, and its capture-script
checksum. The downloadable `capture_rope_example.py` is self-contained and uses
the same pinned dependencies in `requirements-traces.txt`. Model weights are
downloaded to the caller's cache and are not included in this example file.
Reproduction downloads about 1.5 GB on the first run.

```sh
TOKENIZERS_PARALLELISM=false .trace-venv/bin/python scripts/capture_rope_example.py \
  --cache-dir .hf-cache
python3 scripts/capture_rope_example.py --validate
```

## Masked diffusion language model: LLaDA-8B-Instruct

`diffusion-lm/` holds one recorded generation by
[LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/tree/08b83a6feb34df1a6011b80c3c00c7563e963b07),
pinned to revision `08b83a6feb34df1a6011b80c3c00c7563e963b07` (MIT license). The four files total about
2.07 MB: `manifest.json` (16 KB), `timeline.f32` (118 KB) and `step-01.f32` / `step-24.f32` (967 KB each).
Each binary file is little-endian float32; the manifest lists every tensor's offset and shape, and the
file's byte size and SHA-256, which the page checks before use.

LLaDA writes an answer by unmasking, not left to right. The chat-formatted prompt
`Why is the sky blue? Answer in one sentence.` (24 tokens) is followed by 24 `[MASK]` tokens
(`<|mdm_mask|>`, id 126336). Each of the 24 steps runs one forward pass over all 48 positions with
**bidirectional** attention (no causal mask; `[MASK]` positions attend everywhere and are attended to).
The model predicts every masked position at once. The low-confidence remasking rule keeps the one
prediction with the highest probability and leaves the other masked positions as `[MASK]`. The
recorded answer is `The sky appears blue because the Earth's atmosphere scatters blue light more than any
other color in the spectrum.`, followed by `<|eot_id|>` and one `<|endoftext|>`. The order is not left to
right: the closing `.`, `<|eot_id|>` and `<|endoftext|>` are fixed at steps 5–7, and ` the` (slot 6) is
kept at step 9 with confidence 0.332.

The sampler follows
[ML-GSAI/LLaDA `generate.py`](https://github.com/ML-GSAI/LLaDA/blob/f2683fd9cd10721f2853bafa52f8572ae8155d0c/generate.py)
with `steps=24`, `gen_length=24`, `block_length=24`, `temperature=0` (argmax, no Gumbel noise),
`cfg_scale=0`, `remasking='low_confidence'` and seed 0. Captured on October 24, 2026 at 20:32 UTC on Apple
MPS in bfloat16 (as LLaDA generates), Python 3.12.12, PyTorch 2.14.1, Transformers 4.57.6. The manifest
records the six weight shards' SHA-256, the SHA-256 of the two remote-code files `trust_remote_code`
executes (the script refuses other versions), package versions and the capture script's SHA-256.

- `timeline` (JSON) and `timeline.f32`: for every step, the answer slots that were masked, the slot the
  rule kept, and at every masked slot the argmax token (`predicted`), the sampler's confidence
  (`confidence`: generate.py's `softmax(logits)` at that token, in bfloat16), and the 16 most likely
  tokens (`candidateIds`, `candidateLogits`, `candidateProbabilities` from a float32 softmax of the same
  logits over all 126,464 vocabulary entries, plus `logSumExp`). Unmasked slots hold −1.
- `step-01.f32` and `step-24.f32`: the forward passes of the first step (all 24 slots masked) and the last
  (one slot left), each with its input token ids in the manifest. Block 1 is recorded for every position:
  the embedding lookup, RMSNorm 1, per-head Q and K before and after RoPE and V (32 heads × 128), the raw
  `QKᵀ` scores and attention probabilities for all 32 heads (48 × 48 each), each head's output, the
  attention output projection, the residual, RMSNorm 2, the SwiGLU gate, up projection, SiLU(gate),
  their product, the down projection and the block output; also the final RMSNorm output. Vectors are
  stored as their first 8 coordinates (`name`) and the norm over all coordinates (`nameNorm`; 4,096,
  12,288 or 128). Blocks 2–32 are recorded only as per-position norms of the block input, attention
  output, MLP output and block output.

Validation, stored in `validation`:

- Forward hooks only copy tensors. Rerunning each recorded step without hooks gives identical logits
  (max error 0), and rerunning the whole sampler without hooks gives the same answer and the same kept
  slot at every step.
- Replaying each of the 32 blocks on its captured input reproduces its captured output exactly. The
  captured block-1 output driven through blocks 2–32, the final RMSNorm and the LM head reproduces the
  model's logits exactly, as does the captured block-32 output. Residuals, SwiGLU, the embedding lookup
  and RoPE (recomputed in float32 from the projections with the model's own sin/cos) also match exactly.
- The attention probabilities are `softmax(QKᵀ/√128)` computed in float32 from the captured bfloat16 Q
  and K; rows sum to 1 within 2.4e-7. Multiplied by the captured V they reproduce the model's own
  attention output within 0.0072 (its largest coordinate is 3.42): bfloat16 rounding inside the model's
  fused attention, which never exposes its probabilities.
- The sampler's bfloat16 confidence and the float32 probability of the same token differ by at most
  0.0021. In 14 masked slots two tokens had exactly equal bfloat16 logits; the sampler's argmax (the
  lower token id) is listed first. At step 2 slots 2 (` sky`) and 4 (` blue`) had the same confidence,
  0.984375; `torch.topk` kept ` sky` and ` blue` was kept at step 3. The manifest counts this as
  `remaskingTieSteps: 1`.

Reproduce (downloads about 16 GB of weights into the cache; needs about 17 GB of free memory):

```sh
HF_HOME=/private/tmp/explainer-hf-cache python scripts/capture-diffusion-lm.py \
  --cache-dir /private/tmp/explainer-hf-cache/hub
```

`npm test` checks the artifact: hashes, shapes, every attention row, the timeline's remasking choices
against the recorded confidences, and that the kept tokens spell out the recorded answer.

## Encoder-only recording · ModernBERT-base fill-mask

`encoder-only/` holds the complete fill-mask forward pass of
[ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base/tree/8949b909ec900327062f0ebf497f51aef5e6f0c8),
pinned to revision `8949b909ec900327062f0ebf497f51aef5e6f0c8` (Apache-2.0, weights SHA-256
`340ac08b…b2023a9`). It was captured on October 24, 2026 at 18:42 UTC on CPU in float32, with
Python 3.12.12, PyTorch 2.14.1, Transformers 4.57.6 and eager attention. `manifest.json` records
the full package versions and the SHA-256 of the capture script and of the installed modeling file.

| Example   | Exact sentence                      | Tokens | [MASK] at | Top prediction |
| --------- | ----------------------------------- | -----: | --------: | -------------- |
| `capital` | `The capital of France is [MASK].`  |      9 |         6 | ` Paris` 92.3% |
| `barked`  | `The [MASK] barked at the mailman.` |     11 |         2 | ` dog` 80.1%   |

In the second sentence the clue comes after the mask, so only bidirectional attention can use it.

Each example has `model.f32` and `block-00.f32` … `block-21.f32`: lossless little-endian float32
tensors whose names, offsets and shapes are listed in the manifest, which also checks every file's
size and SHA-256. Every one of the 12 heads and every position is stored at all 22 blocks:

- `input`, `norm1` (attention input; block 1 has no attention LayerNorm, so it equals `input`),
  `queryBefore` / `keyBefore` (the Wqkv output split into heads), `query` / `key` (after RoPE,
  recorded from the installed `apply_rotary_pos_emb`), `value` (never rotated), `scores` (rotated
  Q·Kᵀ × 1/8), `mask` (the additive mask the model received), `attention`, `context` (A·V per
  head), `attentionOutput` (after Wo), `residualAfterAttention`, `norm2`, the GeGLU's `up` (first
  half of Wi, the GELU input), `gate` (second half), `gelu`, `product` and `down`, and `output`.
- `model.f32`: the token-embedding lookup, the embedding LayerNorm output x₀, the final
  LayerNorm output, the masked-LM head's dense, GELU and LayerNorm outputs, and all 50,368 logits
  at the [MASK] position. The decoder is tied to the token embeddings and adds a bias.

ModernBERT alternates attention types. Blocks 1, 4, 7, 10, 13, 16, 19 and 22 attend globally
with RoPE base 160,000; the other 14 use a 128-token sliding window (±64 positions) with RoPE
base 10,000. Both sentences are much shorter than the window, so the sliding-window mask the
model actually received is all zeros (checked, not assumed) and every block sees every token.

The capture compares a hooked run against an unhooked one (identical logits) and against the SDPA
implementation (largest logit difference 1.9e-4). It then reconstructs, at every block: RoPE in
float64 from the configured base (≤ 2.2e-6), the softmax of scores plus mask (exact), A·V
against Wo's input (exact), both residual additions (exact), the GeGLU split and product (exact),
exact-erf GELU in float64 (≤ 4.4e-7), the block chain, the final LayerNorm and the masked-LM
head (exact). The maxima are stored in the manifest.

```sh
TOKENIZERS_PARALLELISM=false .trace-venv/bin/python scripts/capture-encoder-only.py --cache-dir .hf-cache
python3 scripts/capture-encoder-only.py --validate   # NumPy only: re-checks the stored arithmetic
```

The first run downloads about 600 MB. This capture used the package versions listed above, which
are newer than `requirements-traces.txt`; ModernBERT needs Transformers 4.48 or later.

## Whisper speech recording

`speech/` holds one complete transcription by the pretrained
[OpenAI Whisper base](https://huggingface.co/openai/whisper-base/tree/e37978b90ca9030d5170a5c07aadb050351a65bb)
model (Apache-2.0), revision `e37978b90ca9030d5170a5c07aadb050351a65bb`. Its weights file has
SHA-256 `07cadb9f25677c8d50df603e66a98fbd842cce45047139baeb16e6219a1e807b`. Whisper base has 6 encoder blocks, 6 decoder blocks, width 512, 8 heads
of 64 and 72,593,920 parameters. The recording stores activations only, never parameters.

The input is LibriSpeech dev-clean utterance `1272-135031-0012`, "WHERE IS MY BROTHER NOW" (2.04 s, 16 kHz,
CC BY 4.0; see `licenses/NOTICE.txt`). It is read from the pinned
`hf-internal-testing/librispeech_asr_dummy` parquet (revision `5be91486e11a2d616f4ec5db8d3fd248585ac07a`), and its original FLAC bytes are
written unchanged to `static/architectures/speech-librispeech-1272-135031-0012.flac`. Greedy decoding with the
prompt `<|startoftranscript|><|en|><|transcribe|><|notimestamps|>` gives " Where is my brother now?".

Captured on October 24, 2026 at 20:30 UTC on CPU in float32 (eager attention), Python 3.12.12, PyTorch 2.14.1,
Transformers 4.57.6. One unmodified forward pass runs over the full 30-second input and the generated prefix.
Forward hooks only observe it. A second pass without hooks matches every logit exactly.

| File            | Contents                                                                                                                                                                                                                           |    Size |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------: |
| `manifest.json` | Model, input, provenance, time bins, tokens, decoded vocabulary, validation, tensor offsets                                                                                                                                        | 0.48 MB |
| `frontend.f32`  | Waveform; min/max per 10 ms; full log-mel (80 × 3000); conv1 and conv2 before and after GELU, sinusoid positions and encoder input for the first 110 positions (2.2 s)                                                             |  2.9 MB |
| `encoder.f32`   | All 6 blocks: binned attention per head, block input per bin; final hidden state per bin                                                                                                                                           |  0.2 MB |
| `decoder.f32`   | All 6 blocks at full resolution: Q, K, V, scores, causal attention, cross-attention queries, cross-attention over all 1,500 encoder positions, block inputs and outputs; final hidden state; all 51,865 logits at all 10 positions |  5.8 MB |

What is reduced, and how:

- The encoder always attends over all 1,500 positions (20 ms each). That includes the 1,398 positions after
  the audio ends, which Whisper fills with zeros. Every softmax uses every position.
- For drawing, the positions are grouped into 12 time bins: 11 bins of 10 positions (0.2 s) covering 0–2.2 s,
  then one bin for positions 110–1,499 (2.2–30 s, all padding). Binned attention rows are the **mean over the
  query positions in a bin**. Columns **sum the attention mass over the key positions in a bin**, so every row
  still sums to 1. A bin's vector is the mean of its positions. The decoder's cross-attention bins are exact sums
  of the stored full-resolution rows.
- conv1/conv2 activations are stored for the first 2.2 s only. Their full-length recomputation is checked during
  capture.

Validation, from `manifest.json` (maximum absolute differences; reconstructions run in float64):

| Check                                                                          |                                              Max error |
| ------------------------------------------------------------------------------ | -----------------------------------------------------: |
| Hooked vs unhooked logits                                                      |                                                      0 |
| Log-mel, NumPy vs the torch path the model used                                |                                                 6.8e-6 |
| conv1 / conv2 recomputed from the recorded inputs                              |                                        6.5e-6 / 1.7e-5 |
| GELU after conv1, encoder input = GELU(conv2) + positions                      |                                                  0 / 0 |
| Position table vs the sinusoid formula                                         | 2.44e-4 (table stored in float16: half a float16 step) |
| softmax(q·k / 8) vs returned attention, all 18 attention layers                |                                                 3.9e-6 |
| A·V vs out-projection input                                                    |                                                 5.2e-5 |
| Binned row mass vs 1                                                           |                                                 2.5e-6 |
| Residual adds, feed-forward GELU, decoder embedding, cross-attention key input |                                                      0 |
| Decoder attention to future tokens                                             |                                                      0 |
| LM head = final hidden · token embeddingᵀ (tied)                               |                                                 4.3e-5 |

The raw argmax of the logits matches the greedy token at all 7 text positions. Greedy decoding also applied
the generation config's token suppression.

```sh
uv venv .speech-venv --python 3.12
uv pip install --python .speech-venv/bin/python torch==2.14.1 transformers==4.57.6 huggingface-hub==0.36.2 \
  tokenizers==0.22.2 safetensors==0.8.0 numpy==2.5.4 soundfile==0.14.0 pyarrow==21.0.0
TOKENIZERS_PARALLELISM=false .speech-venv/bin/python scripts/capture-speech.py --cache-dir .hf-cache
node --test tests/speech-full.test.mjs
```

## Time-series recording · Chronos-T5 small

`time-series/` holds one complete probabilistic forecast by
[Chronos-T5 small](https://huggingface.co/amazon/chronos-t5-small/tree/a971ba21945c4f1796b17a91fe69214b5f4ad472)
(Apache-2.0), revision `a971ba21945c4f1796b17a91fe69214b5f4ad472`. Its weights file has SHA-256
`9c8b6fde5300f72b01c173153bf9288fa0a200614275bf0585071ad71a6a3d43`. Chronos is a T5 encoder–decoder (6 + 6 blocks, width 512, 8 heads of
64, ReLU feed-forward of 2048, RMSNorm, relative position bias, 46,154,240 parameters) whose vocabulary is
4,096 value bins. The recording stores activations only, never parameters.

The input is NOAA NCEI's contiguous-U.S. monthly average temperature in °F, a U.S. Government work in the
public domain ([Climate at a Glance: National Time Series](https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series)).
The CSV was downloaded on October 25, 2026 and is kept verbatim as `time-series/ncei-conus-tavg-monthly.csv`
(SHA-256 `6152f89d28d44fff936532ec09c392be2042aef27b47dbfffe77913ba026978f`, January 1895 – September 2026), because NCEI revises recent
months. The context is the 36 months from September 2022 to August 2025; the forecast covers September 2025 to
August 2026. Those 12 months are in the file but are never shown to the model; the page draws them only as
held-out observations.

The tokenizer is the `chronos-forecasting` 2.3.2 package's `MeanScaleUniformBins`, checked against a float64
re-implementation in the capture script and again in `tests/time-series-full.test.mjs`:

- scale s = mean(|x|) over the context = 54.4836 °F; every value is divided by s;
- 4,093 bin centres are spaced evenly from −15 to 15 (width 0.00733, so one bin is 0.40 °F here); the edges are
  the midpoints, with ±10²⁰ at the ends. `torch.bucketize(…, right=True) + 2` gives the token id (0 = PAD, 1 = EOS);
- the 36 months use 31 distinct bins, token ids 2122–2239, and EOS is appended (37 encoder tokens).

The decoder starts from token 0 and samples 12 tokens per path: `ChronosPipeline.predict` with seed 0, 20 paths,
temperature 1, top-k 50, top-p 1, `min_new_tokens = max_new_tokens = 12` (so EOS is blocked). A sampled id maps
back to its bin centre times s. The 10/50/90 % quantiles are `torch.quantile` over the 20 paths, as
`predict_quantiles` computes them. Against the held-out months the median's mean absolute error is 1.77 °F
(repeating the last 12 months gives 1.95 °F), and 9 of the 12 observations fall inside the 10–90 % band.

Why not Mauna Loa CO₂: mean scaling does not subtract the level, so a 36-month CO₂ window (about 416–428 ppm)
lands in only 6 bins of 3.1 ppm each, and the forecast becomes a staircase. The temperature series spans
31 bins and shows the mechanism better.

Captured on October 25, 2026 at 14:59 UTC on CPU in float32 (eager attention), Python 3.12.12, PyTorch 2.14.1,
Transformers 4.57.6, chronos-forecasting 2.3.2. After sampling, one unmodified teacher-forced forward pass runs over
the 20 sampled decoder inputs. Forward hooks only observe it, and a second pass without hooks matches every
logit exactly.

| File                          | Contents                                                                                                                                                                                                                                  |   Size |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -----: |
| `manifest.json`               | Model, series provenance, tokenizer state, every context month (value, scaled value, token id, bin edges, centre), seasonal bins, sampled ids and paths, quantiles, held-out months, validation, tensor offsets                           |  32 KB |
| `frontend.f32`                | All 4,093 bin centres and 4,094 edges, the context and its scaled values, the 20 dequantized paths, quantiles and mean                                                                                                                    |  34 KB |
| `encoder.f32`                 | Block 1 in full (embedding, RMSNorm, Q, K, V, relative bias and buckets, scores, attention, A·V, out-projection, residual, ReLU activations, feed-forward); all 6 blocks: attention, input, output, seasonal bins; final hidden state     | 2.6 MB |
| `decoder.f32`                 | All 6 blocks for all 20 paths: causal self-attention, cross-attention over the 37 tokens and its seasonal bins, block-input previews (8 coordinates) and norms; block 1 in full for path 1; all 4,096 logits at all 12 steps of all paths | 8.0 MB |
| `ncei-conus-tavg-monthly.csv` | The NOAA NCEI source file, byte for byte                                                                                                                                                                                                  |  20 KB |

What is reduced, and how:

- Every softmax runs over all 37 encoder tokens. For drawing, the encoder is grouped into 13 bins: the 12
  meteorological seasons of the context (Sep–Nov, Dec–Feb, Mar–May, Jun–Aug; three tokens each) and EOS alone.
  Binned attention rows are the **mean over the query tokens in a bin**; columns **sum the attention mass over
  the key tokens in a bin**, so every row still sums to 1. A bin's vector is the mean of its tokens. The page's
  front-end card draws every month and every token id individually.
- The decoder's cross-attention bins are exact sums of the stored full rows. Decoder block-input strips use the
  first 8 of 512 coordinates.

Validation, from `manifest.json` (maximum absolute differences; reconstructions run in float64):

| Check                                                                          |                               Max error |
| ------------------------------------------------------------------------------ | --------------------------------------: |
| Token ids, package vs float64 re-implementation                                |                        0 of 37 mismatch |
| Scale / bin centres / bin edges, float64 vs the package's float32              |                2.9e-6 / 4.8e-7 / 7.2e-7 |
| Sample paths and quantiles, recording vs `ChronosPipeline.predict` (same seed) |                                   0 / 0 |
| Every sampled id in the top 50 of its step; EOS or PAD sampled                 |                       all 240 / 0 times |
| Hooked vs unhooked logits                                                      |                                       0 |
| Teacher-forced logits vs the scores `generate()` sampled from (KV cache)       |      1.4e-5 (same argmax at every step) |
| RMSNorm recomputed (all 30 norms)                                              |                                  5.4e-7 |
| softmax(q·kᵀ + relative bias) vs returned attention, all 18 attention layers   |                                  2.6e-7 |
| A·V vs out-projection input; row sums vs 1                                     |                         3.0e-7 / 2.8e-7 |
| Relative bias from the bucket formula vs `compute_bias`; reused by blocks 2–6  |                                   0 / 0 |
| Residual adds, ReLU, embedding lookup, cross-attention key input               |                                       0 |
| Feed-forward down projection recomputed                                        | 2.9e-4 (4.0e-7 relative to its largest) |
| Decoder attention to future tokens                                             |                                       0 |
| LM head = (final hidden · 512^−½) · token embeddingᵀ (tied)                    |                                  2.0e-5 |
| Binned row mass vs 1                                                           |                                  2.1e-7 |

```sh
uv venv .time-series-venv --python 3.12
uv pip install --python .time-series-venv/bin/python torch==2.14.1 transformers==4.57.6 huggingface-hub==0.36.0 \
  numpy==2.5.4 safetensors==0.8.0 accelerate==1.15.0 chronos-forecasting==2.3.2
.time-series-venv/bin/python scripts/capture-time-series.py --cache-dir .hf-cache
node --test tests/time-series-full.test.mjs
```
