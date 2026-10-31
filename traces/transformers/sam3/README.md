# Recorded SAM 3 concept segmentation

`manifest.json` and its `.f32` and `.png` files record the pretrained
[facebook/sam3](https://huggingface.co/facebook/sam3/tree/3c879f39826c281e95690f02c7821c4de09afae7)
checkpoint, revision `3c879f39826c281e95690f02c7821c4de09afae7` (`model.safetensors` SHA-256
`6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a`), finding every instance of three
text concepts in one photo. `scripts/architectures/record_sam3.py` made them.

## Licence

The checkpoint is gated on Hugging Face under the **SAM License** (last updated 2025-11-19), copied
unchanged as `LICENSE-SAM.txt` (SHA-256 `bec48f70…7e6448`). It grants a non-exclusive, worldwide,
royalty-free licence to use, reproduce, distribute, copy, modify and make derivative works of the SAM
Materials. Whoever distributes them or their derivatives must include a copy of the agreement, and a
publication of research that uses them must acknowledge it. Use must follow applicable law and trade
controls (no ITAR, military, nuclear, espionage or weapons uses), and must not reverse engineer the
materials. They come as is. These files hold activations, scores, boxes and masks computed by the model,
never the checkpoint; the licence travels with them anyway, and the page credits the SAM Materials.

## Capture

CPU, float32, eager attention, Python 3.12.12, PyTorch 2.14.1, torchvision 0.29.1 and Transformers
5.21.0 (`Sam3Model` and `Sam3Processor`; the repo's `sam3_video` config, whose `detector_config`
`Sam3Model` reads). The manifest records the package versions, the script's SHA-256, the config,
processor and tokenizer hashes and every file's SHA-256; the browser checks each `.f32` file against
its hash before using it. Image mode only: `Sam3VideoModel`'s tracker is not recorded.

| File                           | Source                                                                             | License       |
| ------------------------------ | ---------------------------------------------------------------------------------- | ------------- |
| `architectures/sam-beagle.jpg` | [Dog beagle](https://commons.wikimedia.org/wiki/File:Dog_beagle.jpg), Jon Sullivan | Public domain |

The photo is the SAM tab's (1000 × 750), so the two tabs can be compared on the same pixels. The
processor **stretches** it to 1008 × 1008 (SAM padded it instead) and normalizes with mean and
standard deviation 0.5. The manifest records the SHA-256 of the resulting `pixel_values`.

## The model

- **Image encoder**, once per image (446.2 M parameters in the ViT, 7.8 M in the neck): 14-pixel patches,
  a 72 × 72 grid of 5,184, 32 blocks 1,024 wide with 16 heads of 64 and an MLP of 4,736 (exact GELU). The
  learned positions cover 24 × 24 cells (the 336-pixel pre-training size) and are tiled 3 × 3; every
  attention also rotates Q and K by the cell's 2-D position (axial RoPE, θ = 10,000). Blocks 8, 16, 24
  and 32 attend over all 5,184 patches, with positions scaled by 24/72 so they stay in the windows'
  range; the other 28 attend inside 3 × 3 windows of 24 × 24, with no padding. An FPN neck makes four
  256-channel maps from the output: 288², 144², 72² and 36².
- **Text encoder**, once per concept (353.7 M): CLIP's 24-block, 1,024-wide causal text transformer
  on the concept padded to 32 tokens, then a 1,024 → 256 projection of **every** token. CLIP's own
  1,024 → 512 projection (0.5 M) is unused.
- **Fusion encoder** (9.5 M): 6 pre-norm layers over the 72² map. Each cell attends to all 5,184 cells
  (fixed sine positions on Q and K), then to the concept's real tokens (the padding masked), then a ReLU
  MLP 256 → 2,048 → 256.
- **DETR decoder** (11.6 M): 200 learned queries, each with a learned box, and one **presence token**
  in front. 6 post-norm layers: self-attention among all 201 (each query's position is an MLP of the
  sine encoding of its current box), attention to the concept's tokens, attention to the fused map with
  a **box bias** (a per-head number from each cell's log-scaled distances to the query's box edges; the
  presence token gets none), and a ReLU MLP. After every layer a box head refines each box, and the
  presence head (LayerNorm, 3-layer MLP) gives one logit for the image, clamped to ±10.
- **Scoring** (1.2 M): each query's logit is its projection · the projection of the mean of the
  concept's tokens (after an MLP and a LayerNorm), ÷ 16, clamped to ±12. A query's **score** is
  σ(query logit) × σ(presence logit); instances are the queries scoring above 0.5.
- **Mask decoder** (2.3 M): the fused map attends to the concept once more, a pixel decoder adds the
  144² and 288² FPN maps on the way up to 288 × 288, and each query's mask is a 3-layer MLP of the
  query · every pixel's embedding. A semantic head (one channel) is computed but not used for instances.
- The geometry encoder (8.1 M), which reads box prompts, is not used. 840.4 M parameters in all.

## What is stored

Per-token and per-patch vectors keep their first 8 coordinates (`…`) plus full-vector L2 norms
(`…Norm`). Recorded patches: four strips of six on the 72 × 72 grid (`strips`), each crossing a window
edge: from the eye into the ear (row 22, columns 43–48), across the left ear (row 30, 19–24), from the
tongue onto the chest (row 52, 44–49) and from the hose onto a leg (row 62, 19–24).

- `encoder.f32`
  - `patch`, `position`, `embedding` (their sum, before the LayerNorm) for the 24 patches.
  - `input`, `norm1`, `attentionOutput`, `residualAfterAttention`, `norm2`, `up`, `activation`, `down`,
    `output`, each `[32 blocks][24][8]` (block 1's input is the embedding after the LayerNorm).
  - `attention` `[32][16 heads][24][6]`: each recorded patch's post-softmax weight on the six patches of
    its own strip. In a windowed block a key in another window is never attended and is stored as 0.
    `attentionMax` `[32][16][24]` is the largest weight in the patch's whole row (576 keys in a windowed
    block, 5,184 in a global one), which the page shades by.
  - `level72` `[24][8]`: the 72² FPN map at the recorded patches. `vitPca` `[72][72][3]`: the first three
    principal components of the ViT output (1,024 channels; 43.9%, 4.4% and 3.6% of its variance) and
    `vitOutputNorm` `[72][72]`.
- `vit-output.png` and `fpn-<size>.png`: the same kind of picture for the ViT output and each FPN map,
  each component scaled min–max to 0–255 for display. They are views of the features, not inputs to
  anything.
- `concept-<i>.f32`, one per concept:
  - `clipEmbedding`, `clipLayers` `[24][T][8]`, `clipFinal`, and `text` `[T][256]` whole: the real tokens
    after the projection (T = 3 for every concept here: ‹start›, the word, ‹end›).
  - Fusion encoder at the 24 patches, `[6 layers][24][8]` each: `fusionInput`, `fusionNorm1`,
    `fusionSelfOutput`, `fusionResidual1`, `fusionNorm2`, `fusionTextOutput`, `fusionResidual2`,
    `fusionNorm3`, `fusionUp`, `fusionActivation`, `fusionDown`, `fusionOutput`. `fusionSelf`
    `[6][8 heads][24][6]` and `fusionSelfMax` as in the encoder; `fusionText` `[6][8][24][T]` whole
    (the padding's weights are exactly 0). `fusionTextMap` `[8][5184][T]`: the last layer's text
    attention for every cell. `fusedPca` `[72][72][3]`: the fused map's first three principal components.
  - Decoder rows: the presence token and the concept's five highest-scoring queries (`shown`). Stages
    `decoderInput`, `decoderSelfOutput`, `decoderNorm1`, `decoderTextOutput`, `decoderNorm2`,
    `decoderImageOutput`, `decoderNorm3`, `decoderUp`, `decoderActivation`, `decoderDown`,
    `decoderOutput`, each `[6][6][8]`; `queryPos` `[6][5][8]`. `decoderSelf` `[6][8][6][6]` with
    `decoderSelfMax` over all 201 keys; `decoderText` `[6][8][6][T]`; `decoderImage` and `decoderBias`
    `[6][8][6][24]` (the weight on, and the box bias at, each recorded patch) with `decoderImageMax` over
    all 5,184; `decoderImageMap` `[8][6][5184]`: the last layer's whole rows.
  - Every query: `logits` `[6][200]` (each layer's states scored), `boxes` `[7][200][4]` (cx, cy, w, h
    in 0 … 1: the learned start, then after each layer), `scores` `[200]`, `presenceLogits` `[6]`.
  - Scoring: `textScored`, `pooledText`, `projText`, `projQueries` `[6][5][8]`; masks: `maskEmbed`
    `[5][8]`.
- `concept-<i>-query-<q>.png`: query q's mask at the photo's 1000 × 750, white with alpha = mask, for the
  five shown queries. It is the processor's post-processing of the model's 288 × 288 logits (sigmoid,
  bilinear resize, > 0.5); for the kept queries it matches the processor's own output pixel for pixel.

## Results

| Concept   | Tokens              | Presence logits, layers 1–6              | Presence | Kept (query: score)    |
| --------- | ------------------- | ---------------------------------------- | -------: | ---------------------- |
| “dog”     | 49406, 1929, 49407  | 4.75, 5.09, 4.60, 4.73, 5.09, 4.83       |   0.9921 | 144: 0.982             |
| “ear”     | 49406, 8373, 49407  | 6.72, 6.93, 7.29, 7.84, 7.87, 7.22       |   0.9993 | 104: 0.925, 121: 0.882 |
| “bicycle” | 49406, 11652, 49407 | −7.15, −6.51, −7.03, −7.23, −7.19, −7.03 |  0.00089 | none                   |

- “dog”: query 144's box is (267, 101)–(1002, 749) px, almost the box the SAM tab prompts with
  ((265, 95)–(1000, 750)); its mask covers 48.7% of the photo. The next query scores 0.116.
- “ear”: two instances, one per ear: query 104 at (267, 134)–(365, 453) and query 121 at
  (537, 101)–(679, 390).
- “bicycle” is not in the photo. Query 176 still rates itself σ(0.004) = 0.501 on its own, with a box
  over the bottom strip of the photo where the coiled hose lies; the presence head's 0.00089 multiplies
  every query, so its score is 4.4 × 10⁻⁴ and nothing is kept. This is the presence token's job: the
  queries say where an instance would be, the presence token whether there is one.

## Validation

Forward hooks only observe (their return values are dropped). The capture checks, and stores the maxima
in `validation`:

- The hooked image encoder equals the unhooked one (0); every concept's hooked forward equals an
  unhooked one (0); the full model on the pixels equals the image encoder run once plus the detector
  on its features (0).
- Every recorded image-encoder row is recomputed from the block's own Q and K projections and rotary
  angles (5.4 × 10⁻⁶); every recorded attention row sums to 1 (5.6 × 10⁻⁶); the padding gets exactly 0.
- Patch + tiled position, every residual addition, GELU and ReLU, every decoder residual + LayerNorm,
  the decoder's output LayerNorm, the presence head, the box refinement, the query logits (rebuilt from
  `projQueries` · `projText` ÷ 16) and the masks (rebuilt as mask embedding · pixel embedding) equal the
  model's own tensors (0).
- Scores, boxes in pixels and masks equal the processor's `post_process_instance_segmentation` with
  threshold 0.5 (0 mismatched pixels).

`pytorch/sam3.py`, the tab's PyTorch drawer, is checked section by section against `Sam3Model` on these
weights, this photo and these concepts (`pytorch/checks/sam3.py`, `pytorch/results/sam3.json`).
