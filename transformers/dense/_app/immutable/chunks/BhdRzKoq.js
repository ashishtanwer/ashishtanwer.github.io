const n=`# Shap-E (openai/shap-e), the text-to-3D model in the Shap-E tab, written as plain PyTorch blocks. A transformer
# (the "prior") denoises a 1024 × 1024 latent with Heun steps; the latent is then projected, row by row, into the first
# four weight matrices of an MLP that maps a 3D point to a signed distance, densities and colours. Rendering that MLP
# with a NeRF, or meshing its signed distance, gives the 3D object.
# Each "# %% [id]" section is one block of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    width: int = 1024  # the prior's width, and the length of one latent row
    rows: int = 1024  # latent rows: 4 × 256, one per row of the MLP's first four weight matrices
    n_heads: int = 16  # 1024 / 16 = 64 dimensions per head
    n_layers: int = 24
    time_width: int = 4096  # hidden width of the timestep MLP
    text_width: int = 768  # CLIP ViT-L/14's projected text embedding
    out_width: int = 2048  # x̂₀ (1024) and a variance the sampler drops (1024)
    guidance: float = 15.0
    train_steps: int = 1024
    steps: int = 64
    rho: float = 7.0  # Karras spacing of the σs
    param_shapes: tuple = ((256, 93), (256, 256), (256, 256), (256, 256))  # W₀…W₃ of the MLP
    hidden: int = 256
    n_out: int = 12  # signed distance, 2 densities, 3 RGB colours
    pos_octaves: int = 15  # x, y, z and sin/cos of 2⁰…2¹⁴ times each: 3 + 90 = 93 numbers
    dir_octaves: int = 8  # the view direction: 3 + 48 = 51 numbers (zeros when rendering without one)
    direction_at: int = 4  # the layer that also reads the direction


# %% [text-encoder] CLIP's text embedding → the prior's condition
def condition(text_embeds, width=768):
    """CLIP ViT-L/14's projected embedding at <end> (text_embeds of CLIPTextModelWithProjection), scaled to length
    √768 so its numbers have about unit variance. For guidance the condition is all zeros, not an empty prompt."""
    c = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    return math.sqrt(width) * c  # (batch, 768)


def conditions(text_embeds, width=768):
    """The batch the sampler runs: [∅ = zeros, c]."""
    c = condition(text_embeds, width)
    return torch.cat([torch.zeros_like(c), c])


# %% [timestep-embedding] Timestep embedding
def sinusoid(t, dim):
    """cos then sin of t at dim/2 frequencies 10000^(−i/(dim/2)); t may be fractional (Karras σs map between steps)."""
    half = dim // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    angles = t[:, None].float() * frequencies[None]
    return torch.cat([angles.cos(), angles.sin()], dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.linear_1 = nn.Linear(c.width, c.time_width)
        self.linear_2 = nn.Linear(c.time_width, c.width)

    def forward(self, t):  # (batch,) → (batch, 1024): one token of the sequence
        return self.linear_2(F.gelu(self.linear_1(sinusoid(t, self.linear_1.in_features))))


# %% [attention] Self-attention
class Attention(nn.Module):
    """Every token attends to every token (no mask): the condition, the timestep and all 1024 latent rows."""

    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        self.q = nn.Linear(c.width, c.width)
        self.k = nn.Linear(c.width, c.width)
        self.v = nn.Linear(c.width, c.width)
        self.out = nn.Linear(c.width, c.width)

    def forward(self, x):  # (batch, tokens, width)
        b, n, d = x.shape
        q, k, v = (proj(x).view(b, n, self.n_heads, -1).transpose(1, 2) for proj in (self.q, self.k, self.v))
        weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d // self.n_heads), dim=-1)  # (b, h, n, n)
        return self.out((weights @ v).transpose(1, 2).reshape(b, n, d))


# %% [mlp] MLP
class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.width, 4 * c.width)
        self.down = nn.Linear(4 * c.width, c.width)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


# %% [block] Prior block
class Block(nn.Module):
    """Pre-norm: LayerNorm → attention → add; LayerNorm → MLP → add. The prior has no cross-attention: the
    condition and the timestep are tokens in the same sequence."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(c.width)
        self.attention = Attention(c)
        self.norm2 = nn.LayerNorm(c.width)
        self.mlp = MLP(c)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


# %% [heun] Karras σs and the Heun step
def exp_alphas_cumprod(train_steps=1024):
    """Shap-E's training schedule: ᾱ(s) = exp(−12 s) on s ∈ [0, 1], as betas β = 1 − ᾱ(s + 1/T)/ᾱ(s)."""
    s = torch.arange(train_steps + 1, dtype=torch.float64) / train_steps
    alpha_bar = torch.exp(-12 * s)
    betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999).float()
    return torch.cumprod(1 - betas, dim=0)


def karras_sigmas(steps=64, train_steps=1024, rho=7.0):
    """The sampler's σs (noise per unit of signal, σ = √((1 − ᾱ)/ᾱ)) from the largest training σ to the smallest,
    evenly spaced in σ^(1/ρ), then 0; and the fractional training step t each σ maps back to, which the prior reads."""
    alphas_cumprod = exp_alphas_cumprod(train_steps)
    train_sigmas = ((1 - alphas_cumprod) / alphas_cumprod).sqrt().double()  # (1024,), from 0.11 to 403
    sigma_max, sigma_min = train_sigmas[-1].item(), train_sigmas[0].item()
    ramp = torch.linspace(0, 1, steps, dtype=torch.float64)
    sigmas = (sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    log_train = train_sigmas.log()
    low = ((sigmas.log()[None] - log_train[:, None]) >= 0).cumsum(0).argmax(0).clamp(max=train_steps - 2)
    w = ((log_train[low] - sigmas.log()) / (log_train[low] - log_train[low + 1])).clamp(0, 1)
    t = (1 - w) * low + w * (low + 1)
    return torch.cat([sigmas, torch.zeros(1, dtype=torch.float64)]).float(), t.float()


def heun_step(denoise, x, sigma, sigma_next):
    """One step of Heun's method on dx/dσ = (x − x̂₀(x, σ))/σ: an Euler step to σ_next, the slope again there, and the
    step redone with the mean of the two slopes. The last step (to σ = 0) is plain Euler, which lands on x̂₀."""
    x0 = denoise(x, sigma)
    d = (x - x0) / sigma
    x_euler = x + (sigma_next - sigma) * d
    if sigma_next == 0:
        return x_euler, x0
    d_next = (x_euler - denoise(x_euler, sigma_next)) / sigma_next
    return x + (sigma_next - sigma) * (d + d_next) / 2, x0


# %% [guidance] Classifier-free guidance
def guide(x0_null, x0_text, scale=15.0):
    """The prompt's prediction pushed 15× as far from the zero condition's; the sampler then clamps it to [−1, 1]."""
    return x0_null + scale * (x0_text - x0_null)


# %% [step] One denoising step
def denoiser(prior, conds, t_of_sigma, c: Config = Config()):
    """x̂₀(x, σ) as the sampler sees it: the prior on x / √(σ² + 1) at the σ's training step, for ∅ and the prompt,
    mixed by guidance and clamped to the latent's range [−1, 1]."""

    def denoise(x, sigma):
        scaled = (x / math.sqrt(sigma**2 + 1)).expand(2, -1, -1)
        t = torch.full((2,), float(t_of_sigma(sigma)))
        x0_null, x0_text = prior(scaled, t, conds).chunk(2)
        return guide(x0_null, x0_text, c.guidance).clamp(-1, 1)

    return denoise


# %% [params-proj] Latent → MLP weights
class ChannelsProj(nn.Module):
    """One weight matrix of the MLP from its rows of the latent: row r goes through its own Linear (width → columns),
    a LayerNorm over the columns, and its own bias. 256 latent rows → a 256 × 93 or 256 × 256 matrix."""

    def __init__(self, rows, columns, width):
        super().__init__()
        self.rows, self.columns = rows, columns
        self.proj = nn.Linear(width, rows * columns)
        self.norm = nn.LayerNorm(columns)

    def forward(self, latent_rows):  # (batch, rows, width)
        weight = self.proj.weight.view(self.rows, self.columns, -1)  # each row's own (columns × width) matrix
        h = torch.einsum('brw,rcw->brc', latent_rows, weight)
        return self.norm(h) + self.proj.bias.view(1, self.rows, self.columns)


class ParamsProj(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.projections = nn.ModuleList(ChannelsProj(rows, cols, c.width) for rows, cols in c.param_shapes)

    def forward(self, latent):  # (batch, 1024, 1024) → [W₀ (batch, 256, 93), W₁, W₂, W₃ (batch, 256, 256)]
        out, start = [], 0
        for proj in self.projections:
            out.append(proj(latent[:, start : start + proj.rows]))
            start += proj.rows
        return out


# %% [posenc] Positional encoding
def posenc(x, octaves):
    """NeRF's encoding: the 3 coordinates, then sin(2^k x) for k < octaves (all 3 coordinates per k), then the same
    with cos. 15 octaves: 3 + 45 + 45 = 93 numbers."""
    scales = 2.0 ** torch.arange(octaves, dtype=x.dtype)
    xb = (x[..., None, :] * scales[:, None]).flatten(-2)  # (…, 3 · octaves)
    return torch.cat([x, torch.sin(torch.cat([xb, xb + math.pi / 2], dim=-1))], dim=-1)


# %% [field] The MLP field
class Field(nn.Module):
    """The decoded 3D function. W₀…W₃ come from the latent; every bias and W₄…W₆ are the same for every object.
    SiLU between layers; layer 4 also reads the view direction (zeros here: the renders pass none)."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        d_pos, d_dir = 3 * (1 + 2 * c.pos_octaves), 3 * (1 + 2 * c.dir_octaves)
        widths = [d_pos] + [c.hidden] * 6
        widths_in = [w + (d_dir if i == c.direction_at else 0) for i, w in enumerate(widths)]
        self.mlp = nn.ModuleList(nn.Linear(w, o) for w, o in zip(widths_in, [c.hidden] * 6 + [c.n_out]))

    def forward(self, points, weights):  # points: (n, 3) in [−1, 1]³; weights: [W₀, W₁, W₂, W₃]
        h = posenc(points, self.c.pos_octaves)
        for i, layer in enumerate(self.mlp):
            if i == self.c.direction_at:
                h = torch.cat([h, h.new_zeros(*h.shape[:-1], 3 * (1 + 2 * self.c.dir_octaves))], dim=-1)
            h = F.linear(h, weights[i] if i < len(weights) else layer.weight, layer.bias)
            if i < len(self.mlp) - 1:
                h = F.silu(h)
        return {
            'sdf': torch.tanh(h[..., 0:1]),  # > 0 inside the object
            'density_coarse': F.relu(h[..., 1:2]),
            'density_fine': F.relu(h[..., 2:3]),
            'stf': torch.sigmoid(h[..., 3:6]),  # the mesh's texture colour
            'nerf_coarse': torch.sigmoid(h[..., 6:9]),
            'nerf_fine': torch.sigmoid(h[..., 9:12]),  # the renders' colour
        }


# %% [render] Volume rendering
def composite(ts, t0, t1, density, colour, background):
    """A ray's colour from samples at depths ts (n, 1) inside the box [t0, t1]: each sample owns the interval between
    the midpoints around it; α = 1 − exp(−density · length); light reaches sample i with T_i = exp(−Σ_{j<i}
    density·length); colour = Σ α_i T_i c_i, plus the background behind whatever light gets through."""
    mids = (ts[1:] + ts[:-1]) / 2
    lengths = torch.cat([mids, t1.view(1, 1)]) - torch.cat([t0.view(1, 1), mids])
    optical = density * lengths
    reach = torch.exp(-torch.cat([optical.new_zeros(1, 1), optical.cumsum(0)[:-1]]))
    weights = (1 - torch.exp(-optical)) * reach
    through = torch.exp(-optical.sum(0))
    return (weights * colour).sum(0) + through * background, weights


# %% [mesh] Marching cubes
def marching_cubes(field, cases, masks, low=-1.0, high=1.0):
    """The surface field = 0 of a (n, n, n) grid over [low, high]³, with the 256-case triangle table (cases: (256, 5, 3)
    edge numbers; masks: which of the 5 triangles exist). Each cube's 8 corner signs pick a case; each triangle corner
    sits on a cube edge, where the field crosses 0 by linear interpolation."""
    n = field.shape
    inside = (field > 0).to(torch.uint8)
    code = inside[:-1] | inside[1:] << 1  # corner bits: x, then y, then z
    code = code[:, :-1] | code[:, 1:] << 2
    code = code[:, :, :-1] | code[:, :, 1:] << 4
    idx = torch.stack(torch.meshgrid(*[torch.arange(s - 1) for s in n], indexing='ij'), -1).reshape(-1, 3)
    x, y, z = idx.unbind(-1)
    nx, ny = (n[0] - 1) * n[1] * n[2], n[0] * (n[1] - 1) * n[2]  # edges along x, then y, then z
    ex = lambda i, j, k: i * n[1] * n[2] + j * n[2] + k  # noqa: E731
    ey = lambda i, j, k: nx + i * (n[1] - 1) * n[2] + j * n[2] + k  # noqa: E731
    ez = lambda i, j, k: nx + ny + i * n[1] * (n[2] - 1) + j * (n[2] - 1) + k  # noqa: E731
    edges = torch.stack(
        [ex(x, y, z), ex(x, y + 1, z), ex(x, y, z + 1), ex(x, y + 1, z + 1),
         ey(x, y, z), ey(x + 1, y, z), ey(x, y, z + 1), ey(x + 1, y, z + 1),
         ez(x, y, z), ez(x + 1, y, z), ez(x, y + 1, z), ez(x + 1, y + 1, z)], -1,
    )  # fmt: skip
    case = code.reshape(-1).long()
    triangles = torch.gather(edges, 1, cases[case].reshape(len(case), -1)).reshape(-1, 3)[masks[case].reshape(-1)]
    used, faces = torch.unique(triangles.reshape(-1), return_inverse=True)
    corners = torch.stack(torch.meshgrid(*[torch.arange(s, dtype=field.dtype) for s in n], indexing='ij'), -1)
    midpoints = torch.cat([((corners[:-1] + corners[1:]) / 2).reshape(-1, 3),
                           ((corners[:, :-1] + corners[:, 1:]) / 2).reshape(-1, 3),
                           ((corners[:, :, :-1] + corners[:, :, 1:]) / 2).reshape(-1, 3)])[used]  # fmt: skip
    a, b = midpoints.floor().long(), midpoints.ceil().long()
    fa, fb = field[a[:, 0], a[:, 1], a[:, 2]], field[b[:, 0], b[:, 1], b[:, 2]]
    scale = (high - low) / (torch.tensor(n, dtype=field.dtype) - 1)
    w = (fa / (fa - fb))[:, None]
    verts = w * (b * scale + low) + (1 - w) * (a * scale + low)
    return verts, faces.reshape(-1, 3)


# %% [model] Prior and sampling loop
class Prior(nn.Module):
    """The sequence is [condition, timestep, 1024 latent rows]; the latent rows get learned positions, the first two
    tokens none. Out of the last LayerNorm, each latent row's token gives 2048 numbers: the first 1024 are x̂₀."""

    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.time_embedding = TimestepEmbedding(c)
        self.embedding_proj = nn.Linear(c.text_width, c.width)
        self.proj_in = nn.Linear(c.width, c.width)
        self.positional_embedding = nn.Parameter(torch.zeros(1, c.rows, c.width))
        self.norm_in = nn.LayerNorm(c.width)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm_out = nn.LayerNorm(c.width)
        self.proj_out = nn.Linear(c.width, c.out_width)

    def forward(self, x, t, cond):  # x: (batch, 1024, 1024) scaled latent; t: (batch,); cond: (batch, 768)
        rows = self.proj_in(x) + self.positional_embedding
        tokens = torch.cat([self.embedding_proj(cond)[:, None], self.time_embedding(t)[:, None], rows], dim=1)
        h = self.norm_in(tokens)
        for block in self.blocks:
            h = block(h)
        return self.proj_out(self.norm_out(h)[:, 2:])[..., : self.c.width]  # x̂₀: (batch, 1024, 1024)


@torch.no_grad()
def sample(prior, text_embeds, generator, c: Config = Config()):
    """Pure noise at σ_max → 64 Heun steps → the final latent, which ParamsProj turns into the MLP's weights."""
    sigmas, ts = karras_sigmas(c.steps, c.train_steps, c.rho)
    t_of_sigma = {float(s): t for s, t in zip(sigmas[:-1].tolist(), ts.tolist())}
    denoise = denoiser(prior, conditions(text_embeds, c.text_width), lambda s: t_of_sigma[float(s)], c)
    x = torch.randn(1, c.rows * c.width, generator=generator).view(1, c.rows, c.width) * sigmas[0]
    for sigma, sigma_next in zip(sigmas[:-1].tolist(), sigmas[1:].tolist()):
        x, _ = heun_step(denoise, x, sigma, sigma_next)
    return x
`;export{n as default};
