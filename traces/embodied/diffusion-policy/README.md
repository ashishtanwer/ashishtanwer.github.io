# Recorded Diffusion Policy in gym-pusht

`manifest.json` and its files are recorded from
[lerobot/diffusion_pusht](https://huggingface.co/lerobot/diffusion_pusht/tree/84a7c23178445c6bbf7e1a884ff497017910f653)
(Chi et al., [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)), revision
`84a7c23178445c6bbf7e1a884ff497017910f653`, Apache-2.0, by `scripts/embodied/record_diffusion_policy.py`
through LeRobot 0.6.1's own `DiffusionPolicy.select_action` (MPS, float32). The files hold activations and
the episode, never the weights.

## Inputs

- `gym_pusht/PushT-v0` (gym-pusht 0.1.8, pymunk 6.11.1) from the start state gym-pusht 0.1.5 drew for seed
  1002, the version the checkpoint's `eval_info.json` was made with (0.1.8 draws a different state for the
  same seed); it is passed as `reset_to_state`.
- Sampling noise from `torch.Generator("cpu").manual_seed(0)`: x_T is drawn by the recorder and passed to
  `select_action`, every DDPM step's z by the scheduler from the same generator. The first noise seed tried
  succeeded: the T block is on target (coverage 0.964) after 136 steps, 17 plans.

## Files

| File               | What it holds                                                                                                                                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `episode.f32`      | every step's agent position, block pose, action sent, reward and coverage                                                                                                                                     |
| `replans.f32`      | every plan: x_T and all 100 DDPM steps' x_t, ε̂ and z, the 16 actions, the 32 spatial-softmax keypoints of both images and their 3 × 3 softmax maps, the 64 image features, the 132-number global conditioning |
| `depth.f32`        | plan 4, all 100 steps: the RMS of each GroupNorm group of every U-Net block, and the first FiLM residual block stage by stage (4 channels, the RMS of all 512, γ and β)                                       |
| `schedule.f32`     | β_t and ᾱ_t (squared cosine, 100 steps), the timestep sinusoid and embedding of every t                                                                                                                       |
| `frames.jpg`       | the board at every step                                                                                                                                                                                       |
| `observations.png` | the two 96 × 96 images each plan read                                                                                                                                                                         |

Each DDPM step is checked in float64 against x\_{t−1} = c₀ · clip(x̂₀) + cₜ · x_t + σ_t · z with the z redrawn from
the generator state the scheduler used, the FiLM block is rebuilt from its weights, and the actions played are
checked against the ones the environment received. `validation` in the manifest lists every check's largest
error.
