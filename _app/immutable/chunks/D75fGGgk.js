const e=`# Linear probes (Alain & Bengio 2016, arXiv:1610.01644) on Qwen2.5-0.5B-Instruct: read the residual stream at the
# last token of each Amazon review, at the embeddings and after each of the 24 blocks, and fit one logistic regression
# per point to tell positive from negative reviews. A probe that reads the concept well on reviews it never saw says
# the concept is linearly present at that point; it does not say the model uses it.
from dataclasses import dataclass

import torch


@dataclass
class Config:
    layers: int = 24
    hidden: int = 896
    points: int = 25  # the token embeddings, then the output of each block
    C: float = 1.0  # inverse L2 strength, as in scikit-learn's LogisticRegression
    train_per_class: int = 200
    test_per_class: int = 100


# %% [residual] The residual stream at every point
def residual_stream(model, input_ids):
    """(points, tokens, hidden): what flows into block 0, then what leaves each block. Hooks only read."""
    seen = []
    keep = lambda module, args, output: seen.append(output[0] if isinstance(output, tuple) else output)  # noqa: E731
    first = model.model.layers[0].register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    handles = [first] + [block.register_forward_hook(keep) for block in model.model.layers]
    with torch.no_grad():
        model(input_ids[None])
    for handle in handles:
        handle.remove()
    return torch.stack([s[0] for s in seen])


# %% [read] One vector per review: its last token
def read_reviews(model, tokenizer, texts):
    """(points, reviews, hidden). Raw text, no chat template; one review at a time, so no padding."""
    vectors = [residual_stream(model, torch.tensor(tokenizer(text).input_ids))[:, -1] for text in texts]
    return torch.stack(vectors, dim=1).double()


# %% [standardize] Standardise with the training split
def standardizer(train):
    """Mean and standard deviation of every dimension over the training reviews (one point)."""
    return train.mean(0), train.std(0, unbiased=False)


# %% [probe] Fit a logistic-regression probe
def fit_probe(z, y, C=1.0, steps=30):
    """Minimise C·Σ log(1 + exp(−s·(w·z + b))) + ½‖w‖² (s = ±1, b not penalised) by Newton's method.
    z: (reviews, hidden) standardised features, y: (reviews,) labels 0/1. Returns w (hidden,), b."""
    n, d = z.shape
    Z = torch.cat([z, torch.ones(n, 1, dtype=z.dtype)], 1)  # the bias as one more feature
    theta = torch.zeros(d + 1, dtype=z.dtype)
    ridge = torch.ones(d + 1, dtype=z.dtype)
    ridge[-1] = 0
    for _ in range(steps):
        p = torch.sigmoid(Z @ theta)
        grad = C * Z.T @ (p - y) + ridge * theta
        hess = C * (Z.T * (p * (1 - p))) @ Z + torch.diag(ridge)
        theta = theta - torch.linalg.solve(hess, grad)
    return theta[:-1], theta[-1]


# %% [predict] The probe's logit, in the residual stream's own basis
def to_residual_basis(w, b, mean, std):
    """logit = w·(x − mean)/std + b = w_raw·x + b_raw, so the probe reads x directly."""
    w_raw = w / std
    return w_raw, b - (w_raw * mean).sum()


def probe_logit(x, w_raw, b_raw):
    return x @ w_raw + b_raw  # > 0: positive review; σ(logit) = P(positive)


# %% [mean-diff] A probe without training: the class-mean difference
def mean_difference(train, y):
    """Direction = mean of positive reviews − mean of negative ones; threshold halfway between the class means."""
    pos, neg = train[y == 1].mean(0), train[y == 0].mean(0)
    direction = pos - neg
    return direction, 0.5 * (pos + neg) @ direction  # score = x·direction − threshold


# %% [accuracy] Accuracy at every point
def accuracy(logits, y):
    return ((logits > 0).double() == y).double().mean(-1)


# %% [model] Every point, trained and held out
def probe_every_point(model, tokenizer, train_texts, train_y, test_texts, test_y, C=1.0):
    X_train = read_reviews(model, tokenizer, train_texts)
    X_test = read_reviews(model, tokenizer, test_texts)
    rows = []
    for x_train, x_test in zip(X_train, X_test):  # one point at a time
        mean, std = standardizer(x_train)
        w, b = fit_probe((x_train - mean) / std, train_y, C)
        w_raw, b_raw = to_residual_basis(w, b, mean, std)
        direction, threshold = mean_difference(x_train, train_y)
        rows.append({
            "train": accuracy(probe_logit(x_train, w_raw, b_raw), train_y),
            "test": accuracy(probe_logit(x_test, w_raw, b_raw), test_y),
            "mean_diff_test": accuracy(x_test @ direction - threshold, test_y),
            "w_raw": w_raw, "b_raw": b_raw,
        })
    return rows
`;export{e as default};
