# Recorded ACT in gym-aloha

`manifest.json` and its files are recorded from
[lerobot/act_aloha_sim_transfer_cube_human](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human/tree/ba73b2766f1371cdc133ca4efb97eb090d744625)
(Zhao et al., [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)), revision
`ba73b2766f1371cdc133ca4efb97eb090d744625`, Apache-2.0, by `scripts/embodied/record_act.py` through LeRobot
0.6.1's own `ACTPolicy` (CPU, float32). The files hold activations and episodes, never the weights.

## Inputs

- `gym_aloha/AlohaTransferCube-v0` (gym-aloha 0.1.5, MuJoCo 3.8.1), seeds 1001 (handed over at step 236) and
  1000 (the right gripper touches the cube but never lifts it; 400 steps). Both outcomes, and those of seeds
  1000–1019, are the same as in the checkpoint's own `eval_info.json`.
- The CVAE encoder's z on episode 0 of
  [lerobot/aloha_sim_transfer_cube_human](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human/tree/6a43d500f101255823a9d2b9dc244eeb01a2cd31)
  (MIT): 20 windows of 100 actions from steps 0, 20, …, 380, with the frame of each window decoded from the
  dataset's video.

## Files

| File                 | What it holds                                                                                                                                                                                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `seedN-steps.f32`    | every step's joint positions, the action the environment received and its reward; for seed 1001 also the chunk the model returns at every step (queried, not played) and LeRobot's `ACTTemporalEnsembler` over those chunks                                        |
| `seedN-chunkC.f32`   | one query: the joints read, RMS maps of ResNet-18's four layers, 4 layer-4 channels, the image tokens, the joint token, the encoder's attention rows of the z and joint tokens (every layer and head), the decoder's self- and cross-attention, the 100 × 14 chunk |
| `seedN-frames.jpg`   | every 4th camera frame of the episode                                                                                                                                                                                                                              |
| `seedN-inputs.jpg`   | the camera image each chunk was predicted from                                                                                                                                                                                                                     |
| `demo.f32`, `demo-…` | the demonstration's joints and actions, μ and log σ² of z for each window, the decoder's chunk with z = 0 and with z = μ, and the window frames                                                                                                                    |
| `common.f32`         | the z token's bias, 4 channels of the 2-D position embedding, and the checkpoint's normalisation statistics                                                                                                                                                        |

`nn.MultiheadAttention` returns only the mean of its heads; every head is rebuilt in float64 from the captured
inputs and checked against that mean and against the module's output. `validation` in the manifest lists every
check's largest error. On the demonstration the CVAE encoder's posterior has collapsed to the prior
(|μ| ≤ 0.0031, σ ≈ 1, KL ≤ 4e-5), so decoding with z = μ changes the chunk by at most 1.1e-5.
