# Recorded gpt-oss-20b · learned sinks, sliding windows, MXFP4 experts

`manifest.json` and `gpt-oss.f32` are one run of the released
[openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee)
checkpoint, revision `6cee5e81ee83917806bbde320786a8fb61efebee`, Apache-2.0 (see
`../../licenses/Apache-2.0.txt`; the repository's usage policy reads "By using OpenAI gpt-oss-20b, you agree to
comply with all applicable law."). The files hold activations, the learned sink logits, the router logits,
32 MXFP4 weights of one expert matrix and the unembedding rows of the 20 most likely next tokens; never the
checkpoint. The manifest records every shard's SHA-256, the tokenizer, config and chat-template hashes, the
package versions and the capture script's hash; the browser checks `gpt-oss.f32` against its hash.

## How a 21B model ran on this Mac

gpt-oss-20b has 20,914,757,184 parameters (4,187,440,704 active per token: 4 of 32 experts). With its experts in
MXFP4 the checkpoint is 13.8 GB; dequantized to bfloat16 it is about 42 GB, more than this Mac's 36 GB. So the
real model runs **one block at a time**, on the CPU in float32 (every MXFP4 and bfloat16 weight is exact in
float32):

1. **Reply.** The harmony prompt below is answered greedily by a lean float32 implementation in the recorder
   (KV cache; each expert dequantized from the memory-mapped checkpoint only when the router picks it): 19
   tokens in 190.5 s, 2,391 expert loads.
2. **Reference pass.** The prompt and the whole reply run through transformers 4.57.6's own
   `GptOssDecoderLayer`, block by block: load the block's weights (its 32 experts dequantized by transformers'
   `convert_moe_packed_tensors`), run it with eager attention, keep what the page shows, free it (67.6 s for 24
   blocks). Every reply token equals this pass's argmax at its position (0 mismatches), and hooks only observe:
   input + attention output + MoE output reproduce each block's output exactly (0).

## The prompt and the trace

The chat template's harmony format, reasoning effort `low`, the date pinned to 2026-10-30, around one user
message: the first paragraph of Arthur Conan Doyle's _A Scandal in Bohemia_ (The Adventures of Sherlock Holmes,
1892, Project Gutenberg ebook 1661, public domain, file SHA-256 `922e2a12…57cd0`) without its last sentence, which
names her again, then "Who is “the woman” in this passage? Answer with her name." The passage names Irene Adler
once, at positions 114–115. 300 prompt tokens: system 0–60, user 61–297, `<|start|>assistant` 298–299.

The model replied
`<|channel|>analysis<|message|>We know it's Irene Adler.<|end|><|start|>assistant<|channel|>final<|message|>Irene Adler<|return|>`
(`reply`, with each step's top 5 from the reference pass). The trace's sequence is the prompt and the reply up to
its first " Irene": 306 tokens, so the last row (305, "it's") predicts " Irene", at 0.911. At that row a sliding
block sees only keys 178–305; the passage's mention is outside its window, and only full blocks can read it.
Full block 22, head 32 puts 0.957 of the last row's attention on "Irene Adler" (`focusMass`).

## What the numbers mean

- `rows`: 14 shown positions: the first token, the user's `<|start|>user`, " Irene Adler", the last key outside and
  the first key inside the last row's window, the end of the prompt, the start of the reply and its last tokens.
- `cells` `[24 blocks][64 heads][14][14]`: attention weights among the shown rows, from the eager attention
  transformers runs (the sink's share already dropped). `cellScores`: q·k/√64 for the same cells (0 where hidden),
  `lse`: log of the full softmax denominator, the sink included, so every weight is exp(score − lse).
- `sinkLogit` `[24][64]`: the learned sink σ of every head. `sinkShare` `[24][64][14]`: the share the sink takes at
  each shown row, exp(σ − lse) = 1 − Σ weights. `sinkMean`: its mean over all 306 queries. `firstMean`: the mean
  weight on token 0 over the queries that can see it. `sinkByQuery` `[24][306]`: the sink's share, mean of heads.
- `reach` `[24][64][306]`: the last row's attention over every key, every block and head. With the sink's share,
  each row sums to 1; in sliding blocks every key before 178 is exactly 0.
- `map` `[24][39][39]`: all queries binned into 8 × 8-token blocks, mean of the 64 heads (a drawing aid).
- `router` `[24][306][32]`: router logits of every token; `experts.chosen` lists the top 4 at the shown rows.
  `expertParts` `[24][14][4][3][8]`: each chosen expert's gate, up and activation (first 8 of 2,880 coordinates),
  `expertPartsNorm` their full norms, `expertNorm` the experts' output norms.
- `input`, `attention`, `moe`, `output` `[24][14][8]` and their `…Norm`: the residual stream at the shown rows.
- `mxfp4.*`: block 12's top expert at the last row (expert 28), its gate row for hidden unit 1,695, weights
  2,337–2,368 (group 74, the group whose products with the row's input add most to that gate): the 16 stored bytes,
  32 codes, scale byte 119 (2⁻⁸), the 32 decoded weights and inputs; the whole matrix's code and scale-exponent
  histograms and how often a group uses ±6 (47%). Dequantizing is exact (max error 0). OpenAI released the experts
  only in MXFP4, so no unquantized expert exists to compare with; MXFP4's rounding error is shown on block 12's
  query projection, stored in bfloat16, rounded with the same 32-weight scheme (scale 2^⌈log2(amax / 6)⌉, nearest
  E2M1 code, ties to even): relative RMS error 12.0%, mean |error| 0.0011 against an RMS weight of 0.0125.
- `candidates`: the 20 largest logits, with softmax over all 201,088 (`logSumExp`); `finalHidden` and
  `candidateRows` re-derive them.

`node --test tests/gpt-oss-trace.test.mjs` re-derives the sink shares, the window masks, the row sums, the top-4
choices, the MXFP4 decoding and the candidate logits from these files.

## Reproduce

```sh
HF_HUB_CACHE=~/.cache/explainer/hf python scripts/architectures/record_gpt_oss.py
```

It needs the pinned checkpoint's three `model-*.safetensors` shards (13.8 GB); each block is loaded, run and freed in turn.
`--reuse` rewrites the trace from the saved capture without the checkpoint.
