const e=`# Gradient boosting for regression, as the Classical ML Explainer's boosting tab runs it: start from the mean, then
# each round fit a small regression tree to the residuals the ensemble has left and add a shrunken copy of it. This is
# a PyTorch reimplementation of scikit-learn's GradientBoostingRegressor with squared loss (its trees: best split by
# Friedman's improvement, thresholds halfway between float32 feature values, depth-first growth); the check compares
# every round's tree and prediction with the recorded scikit-learn run.
from dataclasses import dataclass

import torch


@dataclass
class Config:
    rounds: int = 200
    learning_rate: float = 0.1  # shrinkage ν: each tree adds ν times what it learned
    max_depth: int = 2  # at most 2^depth leaves per tree


# %% [residuals] What each tree is fitted to
def residuals(y, F):
    """For squared loss ½(y − F)², the negative gradient with respect to the prediction F is y − F: each tree is
    fitted to what the ensemble still gets wrong."""
    return y - F


# %% [split] The best split of a node
def best_split(X, r):
    """Over every feature and every gap between neighbouring distinct values, the split that maximizes Friedman's
    improvement n_L·n_R·(mean_L − mean_R)² / (n_L + n_R) of the residuals r. Ties keep the first one found. Returns
    (feature, threshold, improvement) or None if every feature is constant."""
    n = len(r)
    best = None
    for f in range(X.shape[1]):
        order = torch.argsort(X[:, f], stable=True)
        x, s = X[order, f], torch.cumsum(r[order], 0)
        n_left = torch.arange(1, n, dtype=r.dtype)
        sum_left, total = s[:-1], s[-1]
        diff = (total - sum_left) * n_left - sum_left * (n - n_left)  # n_L·n_R·(mean_R − mean_L)
        gain = diff**2 / (n_left * (n - n_left))
        gain[x[1:] <= x[:-1] + 1e-7] = -1  # no threshold between equal values
        k = int(torch.argmax(gain))
        if gain[k] >= 0 and (best is None or gain[k] > best[2]):  # any valid split, even one that gains nothing
            best = (f, float(x[k] / 2 + x[k + 1] / 2), float(gain[k]))
    return best


# %% [tree] A regression tree, grown depth-first
def fit_tree(X, r, depth, nodes=None):
    """Split until max depth or a node's residuals are all equal; a leaf predicts the mean residual of its samples.
    Nodes are numbered in the order they are made (left subtree first), as scikit-learn's tree_ arrays are."""
    nodes = [] if nodes is None else nodes
    node = {'feature': -1, 'threshold': None, 'value': float(r.mean()), 'samples': len(r), 'left': -1, 'right': -1}
    nodes.append(node)
    split = best_split(X, r) if depth > 0 and len(r) >= 2 and float(r.var(unbiased=False)) > 2.2e-16 else None
    if split is not None:
        f, t, _ = split
        go_left = X[:, f] <= t
        node['feature'], node['threshold'] = f, t
        node['left'] = len(nodes)
        fit_tree(X[go_left], r[go_left], depth - 1, nodes)
        node['right'] = len(nodes)
        fit_tree(X[~go_left], r[~go_left], depth - 1, nodes)
    return nodes


def predict_tree(nodes, X):
    out = torch.empty(len(X), dtype=torch.float64)
    for i, x in enumerate(X):
        k = 0
        while nodes[k]['feature'] >= 0:
            k = nodes[k]['left'] if x[nodes[k]['feature']] <= nodes[k]['threshold'] else nodes[k]['right']
        out[i] = nodes[k]['value']
    return out


# %% [boosting] The ensemble, round by round
def boost(X, y, c: Config):
    """F₀ = mean(y); F_m = F_{m−1} + ν·tree_m(x), tree_m fitted to y − F_{m−1}. Returns the trees and the training
    predictions after every round. scikit-learn compares float32 features with float64 thresholds, so X is rounded
    to float32 first."""
    X = X.float().double()
    F = torch.full_like(y, float(y.mean()))
    trees, history = [], [F]
    for _ in range(c.rounds):
        tree = fit_tree(X, residuals(y, F), c.max_depth)
        F = F + c.learning_rate * predict_tree(tree, X)
        trees.append(tree)
        history.append(F)
    return trees, torch.stack(history)
`;export{e as default};
