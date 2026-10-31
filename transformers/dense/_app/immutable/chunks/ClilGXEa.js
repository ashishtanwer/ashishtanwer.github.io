const e=`# Retrieval-augmented generation (Lewis et al. 2020) on SQuAD questions, the RAG tab, as plain PyTorch around three
# Hugging Face models. A bi-encoder (BAAI/bge-small-en-v1.5) turns the question into a vector and the nearest of the
# 2,067 paragraph vectors are retrieved; a cross-encoder (BAAI/bge-reranker-base) reads the question and each
# retrieved paragraph together and scores them again; the best few paragraphs are pasted into the prompt of a small
# instruction model (Qwen/Qwen2.5-0.5B-Instruct), which answers greedily. The same question is also asked with the
# bi-encoder's own top paragraphs (no reranking) and with no paragraphs, and every answer is scored against SQuAD's
# own answer spans.
# The models are passed in (transformers' AutoModel classes); this file is what RAG adds around them. Each "# %% [id]"
# section is one card of the tab.
import re
import string
from collections import Counter
from dataclasses import dataclass

import torch


@dataclass
class Config:
    query_instruction: str = 'Represent this sentence for searching relevant passages: '  # bge's prefix for queries
    max_length: int = 512  # tokens the encoder and the reranker read
    retrieve: int = 20  # paragraphs retrieved by vector similarity
    keep: int = 3  # paragraphs that go into the prompt, after reranking
    max_new_tokens: int = 48
    top: int = 5  # next-token candidates recorded per step


# %% [embed] Embedding the question
@torch.no_grad()
def embed(encoder, tokenizer, text, cfg=Config(), query=True):
    """bge-small's sentence vector: the last layer's [CLS] state, scaled to length 1. Questions get bge's instruction
    prefix; paragraphs are embedded as they are."""
    ids = tokenizer((cfg.query_instruction if query else '') + text, truncation=True, max_length=cfg.max_length,
                    return_tensors='pt')
    cls = encoder(**ids).last_hidden_state[0, 0]
    return torch.nn.functional.normalize(cls, dim=-1)


# %% [retrieve] Retrieving by vector
def retrieve(query, vectors, k):
    """Exact search: squared distance to every paragraph vector (2 − 2 cos for unit vectors), the k nearest.
    The HNSW and IVF-PQ tabs approximate exactly this. Returns ids and distances."""
    d = ((vectors.double() - query.double()) ** 2).sum(-1)
    order = d.argsort(stable=True)[:k]
    return order, d[order]


# %% [rerank] Reranking with a cross-encoder
@torch.no_grad()
def rerank(reranker, tokenizer, question, passages, cfg=Config()):
    """The cross-encoder reads "<s> question </s></s> paragraph </s>" as one sequence, so every token of the question
    attends to every token of the paragraph, and outputs one relevance logit. Scored one pair at a time (no padding).
    Returns the logits and the order, best first."""
    scores = []
    for passage in passages:
        pair = tokenizer(question, passage, truncation='only_second', max_length=cfg.max_length, return_tensors='pt')
        scores.append(reranker(**pair).logits[0, 0])
    scores = torch.stack(scores)
    return scores, scores.argsort(descending=True, stable=True)


# %% [prompt] Building the prompt
def messages(question, passages=None):
    """The user turn: the kept paragraphs, numbered, then the question and the instruction. Without passages the same
    question and instruction alone. Qwen's chat template adds its default system turn around it."""
    ask = f'Question: {question}\\nAnswer with a short phrase.'
    if passages:
        context = '\\n\\n'.join(f'[{i + 1}] {p}' for i, p in enumerate(passages))
        ask = f'Answer the question using these passages.\\n\\n{context}\\n\\n{ask}'
    return [{'role': 'user', 'content': ask}]


def prompt_ids(tokenizer, chat):
    """Token ids of the whole prompt, ending with the assistant header the answer continues from."""
    return tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, return_tensors='pt')[0]


# %% [generate] Greedy answer
@torch.no_grad()
def generate(lm, ids, eos_ids, cfg=Config()):
    """One token at a time: the model's next-token distribution, its most likely token appended, the keys and values
    of the tokens so far kept in a cache. Stops at an end token or after max_new_tokens. Greedy, so no seed is
    involved. Returns the answer ids and, per step, the top candidates and their probabilities."""
    out = lm(input_ids=ids[None], use_cache=True)
    answer, tops = [], []
    for _ in range(cfg.max_new_tokens):
        probs = out.logits[0, -1].float().softmax(-1)
        p, i = probs.topk(cfg.top)
        tops.append((i, p))
        token = int(i[0])
        answer.append(token)
        if token in eos_ids:
            break
        out = lm(input_ids=torch.tensor([[token]]), past_key_values=out.past_key_values, use_cache=True)
    return torch.tensor(answer), tops


# %% [score] Scoring against SQuAD's answers
def normalize(text):
    """SQuAD v1.1's normalization: lower case, no punctuation, no a / an / the, single spaces."""
    text = ''.join(ch for ch in text.lower() if ch not in set(string.punctuation))
    return ' '.join(re.sub(r'\\b(a|an|the)\\b', ' ', text).split())


def score(answer, golds):
    """SQuAD's exact match and token F1, each the best over the crowd's answers, and whether a gold answer appears
    inside the (often longer) generated answer. Returns (exact, f1, contains)."""
    def f1(gold):
        a, g = normalize(answer).split(), normalize(gold).split()
        same = sum((Counter(a) & Counter(g)).values())
        return 0.0 if same == 0 else 2 * same / (len(a) + len(g))

    exact = max(float(normalize(answer) == normalize(g)) for g in golds)
    contains = max(float(f' {normalize(g)} ' in f' {normalize(answer)} ') for g in golds)
    return exact, max(f1(g) for g in golds), contains


# %% [model] The whole pipeline
def rag(question, models, vectors, passages, prompt_passages, cfg=Config()):
    """Embed, retrieve, rerank, keep the best, prompt, answer; and the same question with the bi-encoder's top
    paragraphs as they came (no reranking) and without retrieval.
    models: dict of encoder, encoder_tokenizer, reranker, reranker_tokenizer, lm, lm_tokenizer; vectors: the
    paragraph vectors (stored in float16, like the question's here); passages: each paragraph's text (what the
    reranker reads); prompt_passages: the same with its article's title (what the prompt shows)."""
    q = embed(models['encoder'], models['encoder_tokenizer'], question, cfg).half().float()
    ids, dists = retrieve(q, vectors, cfg.retrieve)
    scores, order = rerank(models['reranker'], models['reranker_tokenizer'], question, [passages[i] for i in ids], cfg)
    kept = ids[order[: cfg.keep]]
    eos = set(models['lm'].generation_config.eos_token_id)
    tok = models['lm_tokenizer']
    with_context = generate(models['lm'], prompt_ids(tok, messages(question, [prompt_passages[i] for i in kept])), eos, cfg)
    vector_only = generate(
        models['lm'], prompt_ids(tok, messages(question, [prompt_passages[i] for i in ids[: cfg.keep]])), eos, cfg
    )
    without = generate(models['lm'], prompt_ids(tok, messages(question)), eos, cfg)
    return {'retrieved': ids, 'distances': dists, 'scores': scores, 'kept': kept, 'answer': with_context,
            'vector_only': vector_only, 'closed_book': without}
`;export{e as default};
