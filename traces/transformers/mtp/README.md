# Recorded MiMo-7B-RL · multi-token prediction

`manifest.json` and `mtp.f32` are one run of the released
[XiaomiMiMo/MiMo-7B-RL](https://huggingface.co/XiaomiMiMo/MiMo-7B-RL/tree/7c7707596b0ef1032ef82970d7a4a484ca457c4d)
checkpoint, revision `7c7707596b0ef1032ef82970d7a4a484ca457c4d`, MIT licence (the model card's `license: mit`; the
repository has no separate licence file, and its "Use Restrictions" notice says it does not change the licence).
The files hold activations, probabilities and token ids, never weights. The manifest records every shard's
SHA-256, the config, tokenizer and `modeling_mimo.py` hashes, the package versions and the capture script's hash;
the browser checks `mtp.f32` against its hash.

## The model and how it ran

MiMo-7B is a Qwen2-style decoder (36 blocks, width 4,096, 32 query and 8 key/value heads, RoPE θ = 640,000,
7,622,619,136 parameters) plus one multi-token-prediction layer (`num_nextn_predict_layers` = 1, 210,790,400
parameters). At position t the MTP layer reads the main model's last block output h_t and the embedding of token
t+1, normalizes each, concatenates them (8,192), projects back to 4,096, runs one transformer block of its own
(attention over the earlier positions' MTP inputs, SwiGLU MLP), normalizes, and the main model's unembedding turns
the result into a distribution for token t+2.

- The main model runs in transformers 4.57.6's own `Qwen2ForCausalLM` (MiMo's main weights use Qwen2's names), in
  bfloat16 on the Mac's GPU (MPS), SDPA attention.
- The MTP layer is `MiMoMTPLayers` from the checkpoint's `modeling_mimo.py` (SHA-256 `10c39e48…7d7030`), rebuilt
  in the recorder from transformers' Qwen2 modules with the same names, so `model.mtp_layers.0.*` loads strictly.
  The released file itself was not executed: running a model repository's own code was refused by this session's
  permission check, and its attention call targets transformers 4.40.1. Eager attention, so its weights are kept.
- As in vLLM's MiMo (`vllm/model_executor/models/mimo.py`: `MiMoModel.forward` returns the last block's output and
  `compute_logits` applies the final norm), the MTP layer reads the main model's state **before** the final norm,
  and it gets a zero token embedding at position 0 (`mimo_mtp.py`). Fed the state after the final norm instead,
  its guesses match less often (`match.postNormMatches`, 271 against 284).

## The prompt and the three runs

GSM8K main test row 0 (`openai/gsm8k` @ `740312add88f781978c0658806c59bc2815b9866`, MIT; gold answer 18) in MiMo's
chat template with the empty system prompt its README recommends: 78 tokens.

1. **Greedy.** Plain greedy decoding, one main-model pass per token, 320 tokens (the cap: the reasoning has not
   reached its answer). 24.3 s.
2. **Teacher-forced pass** over the prompt and that reply: the main head's distribution for t+1 and the MTP layer's
   for t+2 at every position. At the 319 positions from the prompt's last token on, the MTP layer's top guess is
   the token the model itself wrote two positions later 284 times (**89.0%**, `guesses`, `match`).
3. **Speculative decoding** with the MTP layer as a one-token drafter, with KV caches for both: each main pass runs
   the last token and the MTP guess together; the guess is kept when it equals the main model's argmax, which then
   also gives the next token for free. 321 tokens from **169 main passes** (1.90 tokens per pass); 152 of 168
   guesses kept (**90.5%**; MiMo's README reports about 90% with one MTP layer). The first 320 tokens are identical
   to plain greedy decoding. 15.3 s. Wall times are from this one run on a Mac shared with other jobs.

## What the numbers mean

- `rows`: 12 consecutive positions of the reply around its first "=" ("3 + 4 = 7 eggs. So"), 326–337.
- `h` `[12][8]` and `hNorm`: the main model's last block output at those rows (first 8 of 4,096 coordinates, full
  L2 norm). `p` and `e`: h after `hidden_layernorm` and E[t+1] after `token_layernorm`; `x`: after `input_proj`;
  `a`: the MTP attention output; `x2` = x + a; `y`: the MLP output; `out`: after `final_layernorm`. Each `…Norm` is
  the full norm.
- `cells` `[32 heads][12][12]`: the MTP block's attention among the shown rows; `rowsMean` `[12][397]`: each shown
  row's attention over every key, mean of the 32 heads.
- `mainTop`, `mtpTop`: at each shown row, the main head's top 10 for t+1 and the MTP layer's for t+2.
- `speculative.steps`: per main pass, the guess, the MTP and main top 5, whether it was kept, and the emitted tokens.
- `mtpCheckHidden` `[16][4096]` and `mtpCheck.nextTokens`: the main model's state at positions 0–15 and the next
  tokens, so `pytorch/checks/mtp.py` can run the released MTP layer on real inputs.

`node --test tests/mtp-trace.test.mjs` re-scores every guess, re-counts the matches and replays the speculative
steps against the recorded tokens.

## Reproduce

```sh
HF_HUB_CACHE=~/.cache/explainer/hf python scripts/architectures/record_mtp.py
```

It needs the pinned checkpoint (15.7 GB, all of it held in bfloat16 on the GPU). `--reuse` rewrites the trace from the
saved capture without the checkpoint.
