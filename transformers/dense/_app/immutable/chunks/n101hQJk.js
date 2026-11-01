const e=`# Okapi BM25 (Robertson et al., 1994) as Lucene scores it, the "accurate Lucene" variant of Kamphuis et al. (2020) that
# bm25s implements: the Retrieval Explainer's BM25 tab as plain PyTorch. There is no model and no training. A text
# becomes a bag of stemmed words; a paragraph's score for a query is a sum over the query's words of how rare the word
# is in the corpus (IDF) times how often it occurs in the paragraph, saturated by k1 and normalized for the
# paragraph's length by b.
import re
from collections import Counter
from dataclasses import dataclass

import torch

# Lucene's English stop set, bm25s's stopwords="en".
STOPWORDS = (
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of",
    "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with",
)


@dataclass
class Config:
    k1: float = 1.5  # term-frequency saturation (bm25s's default; Lucene's is 1.2)
    b: float = 0.75  # length normalization, 0 = none, 1 = full
    pattern: str = r"(?u)\\b\\w\\w+\\b"  # words of two or more letters or digits
    stopwords: tuple = STOPWORDS


# %% [tokenize] Text to terms
def words(text: str, c: Config):
    """Lowercase, then every run of two or more word characters."""
    return re.findall(c.pattern, text.lower())


def tokenize(text: str, c: Config, stem=lambda word: word):
    """The terms BM25 counts: words minus stopwords, each reduced to its stem. The recorder passes the Snowball English
    stemmer (PyStemmer's Stemmer("english").stemWord), so "historians" and "historian" are one term."""
    return [stem(w) for w in words(text, c) if w not in c.stopwords]


# %% [index] Term frequencies, lengths and IDF
class Index:
    """The inverted index of a tokenized corpus: tf[t, d] counts term t in paragraph d (sparse, terms × docs)."""

    def __init__(self, docs: list[list[str]]):
        self.terms = sorted({t for d in docs for t in d})
        self.vocab = {t: i for i, t in enumerate(self.terms)}
        rows, cols, counts = [], [], []
        for d, terms in enumerate(docs):
            for t, n in Counter(terms).items():
                rows.append(self.vocab[t])
                cols.append(d)
                counts.append(n)
        shape = (len(self.terms), len(docs))
        self.tf = torch.sparse_coo_tensor([rows, cols], counts, shape, dtype=torch.float32,
                                          check_invariants=False).coalesce()
        self.length = torch.tensor([len(d) for d in docs], dtype=torch.float32)  # terms per paragraph
        self.average = self.length.mean()
        self.df = torch.bincount(self.tf.indices()[0], minlength=len(self.terms)).float()  # paragraphs holding t
        self.idf = idf(self.df, len(docs))


def idf(df, n: int):
    """Lucene's IDF: log(1 + (N − df + 0.5) / (df + 0.5)). A word in every paragraph still gets a small positive weight;
    Robertson's original log((N − df + 0.5) / (df + 0.5)) goes negative past N / 2."""
    return torch.log(1 + (n - df + 0.5) / (df + 0.5))


# %% [weights] Saturation and length normalization
def saturate(tf, length, average, k1: float, b: float):
    """tf / (tf + k1 · (1 − b + b · length / average)): 0 for an absent term, rising towards 1 with every repeat. k1
    sets how fast it saturates (k1 = 0: present or not); b how much a longer-than-average paragraph needs extra
    repeats for the same weight. The classic form multiplies by k1 + 1, which scales every score alike."""
    return tf / (tf + k1 * (1 - b + b * length / average))


def weights(index: Index, c: Config):
    """IDF(t) · saturate(tf[t, d]) for every stored (term, paragraph): the precomputed index bm25s searches."""
    t, d = index.tf.indices()
    w = index.idf[t] * saturate(index.tf.values(), index.length[d], index.average, c.k1, c.b)
    return torch.sparse_coo_tensor(index.tf.indices(), w, index.tf.shape, check_invariants=False).coalesce()


# %% [score] One query against every paragraph
def score(index: Index, w, query: list[str]):
    """Σ over the query's terms of IDF(t) · saturate(t, d), for every paragraph d at once: a term the query repeats
    counts again, a term the corpus never saw adds nothing. Returns (docs,) scores."""
    q = torch.zeros(len(index.terms))
    for t in query:
        if t in index.vocab:
            q[index.vocab[t]] += 1
    return torch.sparse.mm(w.t(), q[:, None])[:, 0]  # (docs, terms) @ (terms, 1)


# %% [rank] Ranking, recall and nDCG
def rank(scores, k: int):
    """The k best paragraphs per query, best first: (values, indices). A stable sort, so paragraphs with equal scores
    (many share 0, holding none of the query's terms) keep corpus order."""
    values, order = torch.sort(scores, dim=-1, descending=True, stable=True)
    return values[..., :k], order[..., :k]


def metrics(ranked, relevant, ks=(1, 5, 10, 100)):
    """ranked: (queries, depth) paragraph ids, best first; relevant: (queries,) the one relevant id per query (SQuAD
    has one). Recall@k is the share of queries whose paragraph is in the top k; MRR@10 and nDCG@10 score 1 / rank and
    1 / log2(rank + 1) inside the top 10, else 0."""
    hit = ranked == relevant[:, None]
    found = hit.any(-1)
    rank1 = hit.float().argmax(-1) + 1
    out = {f"recall@{k}": (hit[:, :k].any(-1)).float().mean() for k in ks}
    out["mrr@10"] = torch.where(found & (rank1 <= 10), 1.0 / rank1, 0.0).mean()
    out["ndcg@10"] = torch.where(found & (rank1 <= 10), 1.0 / torch.log2(rank1 + 1.0), 0.0).mean()
    return out
`;export{e as default};
