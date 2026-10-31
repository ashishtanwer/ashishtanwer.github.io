const e=`# word2vec (Mikolov et al. 2013), skip-gram with negative sampling: the Google News vectors
# (fse/word2vec-google-news-300, 3,000,000 words and phrases × 300) and the objective that trained them.
# A word2vec model is one lookup table. Training slides a window over the text: the centre word's vector should
# score high (dot product) against the output vectors of the words around it and low against a few random words.
# After training, words used in similar contexts have vectors pointing the same way, so cosine similarity finds
# neighbours and vector differences carry relations (king − man + woman lands near queen).
# Each "# %% [id]" section is one part of the word2vec tab.
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 3_000_000  # words and phrases (New_York) of the Google News model, most frequent first
    dim: int = 300
    window: int = 5  # largest distance of a context word; each centre draws its own window 1 … window
    negatives: int = 5  # k random "noise" words per (centre, context) pair
    noise_power: float = 0.75  # noise words are drawn with probability ∝ count^0.75
    sample: float = 1e-3  # frequent words are dropped from the text with probability 1 − (√(f/t) + 1)·t/f
    lr: float = 0.025  # SGD step size, decayed linearly towards lr × 1e-4 over training


# %% [lookup] Word → vector
def lookup(vectors, index, words):
    """vectors: (vocab, dim), one row per word; index: word → row. The only layer of the model."""
    return vectors[torch.tensor([index[w] for w in words])]  # (len(words), dim)


# %% [cosine] Nearest neighbours by cosine similarity
def unit(x):
    return x / x.norm(dim=-1, keepdim=True)


def cosine(a, b):
    """cos θ = a·b / (|a| |b|): 1 for the same direction, 0 for orthogonal. Lengths are ignored."""
    return unit(a) @ unit(b).T


def most_similar(vectors, query, k=10, exclude=()):
    """The k rows of \`vectors\` closest to \`query\` by cosine, skipping the row ids in \`exclude\` (the query's own)."""
    sims = unit(vectors) @ unit(query)  # (vocab,)
    sims[list(exclude)] = -torch.inf
    top = sims.topk(k)
    return top.indices, top.values


# %% [analogy] Analogies by vector arithmetic: a is to b as c is to ?
def analogy_target(vectors, a, b, c):
    """b − a + c on unit vectors (gensim's most_similar(positive=[b, c], negative=[a]) averages them; the mean has
    the same direction). man : king :: woman : ?  →  king − man + woman."""
    u = unit(vectors[torch.tensor([a, b, c])])
    return unit((u[1] - u[0] + u[2]) / 3)


def analogy(vectors, a, b, c, k=5):
    """The answer is the nearest word to the target that is not one of the three inputs: without that rule the
    nearest word to king − man + woman is king itself."""
    return most_similar(vectors, analogy_target(vectors, a, b, c), k, exclude=(a, b, c))


# %% [pca] Two principal components of a group of words
def pca_2d(x):
    """x: (n, dim). Centre the rows and keep the two directions of largest variance (the top right-singular vectors);
    the scatter is each word's coordinates along them. Returns coordinates (n, 2), directions (2, dim) and the share
    of the variance each direction explains."""
    mean = x.mean(0)
    centered = x - mean
    _, s, vh = torch.linalg.svd(centered, full_matrices=False)
    directions = vh[:2]
    # SVD's signs are arbitrary; make each direction's largest coordinate positive so the plot is stable
    directions = directions * directions.gather(1, directions.abs().argmax(1, keepdim=True)).sign()
    explained = (s**2 / (s**2).sum())[:2]
    return centered @ directions.T, directions, explained


# %% [skipgram] Skip-gram with negative sampling: one (centre, context) pair's loss
def keep_probability(counts, total, sample=Config.sample):
    """word2vec drops very frequent words ("the", "and") from the text before windowing: a word with frequency
    f = count / total is kept with probability (√(f/t) + 1)·t/f, at most 1 (t = sample)."""
    f = counts / total
    return ((f / sample).sqrt() + 1) * sample / f


def noise_distribution(counts, power=Config.noise_power):
    """Negative words are drawn from the unigram distribution raised to 3/4: rare words get a little more weight."""
    p = counts.double() ** power
    return p / p.sum()


def window_pairs(tokens, position, reduced):
    """Context words of tokens[position]: up to \`reduced\` words on each side (word2vec draws reduced from 1 … window
    for every centre, so near words are used more often than far ones)."""
    lo, hi = max(0, position - reduced), min(len(tokens), position + reduced + 1)
    return [tokens[j] for j in range(lo, hi) if j != position]


def sgns_loss(v_in, u_out, u_neg):
    """v_in: the centre word's input vector (dim,); u_out: the context word's output vector (dim,);
    u_neg: (k, dim) output vectors of the k noise words. −log σ(u_out·v_in) − Σ log σ(−u_neg·v_in)."""
    return -(F.logsigmoid(u_out @ v_in) + F.logsigmoid(-(u_neg @ v_in)).sum())


def sgns_grads(v_in, u_out, u_neg):
    """The loss's gradients in closed form: each output vector gets (σ(score) − label) × v_in, and the centre gets
    the same coefficients times the output vectors (label 1 for the context word, 0 for the noise words)."""
    g_pos = torch.sigmoid(u_out @ v_in) - 1  # () in (−1, 0): pull the context word closer
    g_neg = torch.sigmoid(u_neg @ v_in)  # (k,) in (0, 1): push each noise word away
    grad_v = g_pos * u_out + g_neg @ u_neg
    return grad_v, g_pos * v_in, g_neg[:, None] * v_in


# %% [step] One SGD update of the two tables
def sgns_step(W_in, W_out, centre, context, negatives, lr):
    """W_in: (vocab, dim) input vectors (the word vectors that get published); W_out: (vocab, dim) output vectors,
    used only in training. One plain SGD step on one pair's loss, as word2vec takes one per pair. A noise word drawn
    twice is updated twice. Returns the loss before the step."""
    v_in, u_out, u_neg = W_in[centre], W_out[context], W_out[negatives]
    loss = sgns_loss(v_in, u_out, u_neg)
    grad_v, grad_out, grad_neg = sgns_grads(v_in, u_out, u_neg)
    W_out[context] -= lr * grad_out
    W_out.index_add_(0, negatives, -lr * grad_neg)
    W_in[centre] -= lr * grad_v
    return loss


# %% [model] Training: every pair of every window, many times over the text
def init_tables(vocab_size, dim, generator):
    """word2vec's start: input vectors uniform in ±0.5/dim, output vectors all zero (so every score starts at 0)."""
    W_in = (torch.rand(vocab_size, dim, generator=generator) - 0.5) / dim
    return W_in, torch.zeros(vocab_size, dim)


def train_epoch(W_in, W_out, sentences, keep, noise, c, lr, generator):
    """sentences: lists of word ids; keep: each word's keep probability; noise: the noise distribution. Frequent
    words are subsampled, each centre gets a random window, and each (centre, context) pair takes one SGD step with
    k fresh noise words. Returns the summed loss and the number of pairs."""
    total, pairs = 0.0, 0
    for sentence in sentences:
        kept = [w for w in sentence if torch.rand((), generator=generator) < keep[w]]
        for position, centre in enumerate(kept):
            reduced = int(torch.randint(1, c.window + 1, (), generator=generator))
            for context in window_pairs(kept, position, reduced):
                negatives = torch.multinomial(noise, c.negatives, replacement=True, generator=generator)
                total += float(sgns_step(W_in, W_out, centre, context, negatives, lr))
                pairs += 1
    return total, pairs
`;export{e as default};
