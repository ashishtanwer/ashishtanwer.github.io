const e=`# DistilBERT, soft-target distillation of BERT-base (12 layers) into a 6-layer student: the Post-training Explainer's
# DistilBERT tab as plain PyTorch. The student starts as a copy of some of the teacher's layers, then learns from
# three losses at once: the teacher's softened distribution (T²·KL), the masked word itself (cross-entropy) and the
# direction of the teacher's last hidden states (cosine). As in Hugging Face's examples/research_projects/distillation.
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Config:
    teacher_layers: int = 12
    student_layers: int = 6
    init_layers: tuple = (0, 2, 4, 7, 9, 11)  # scripts/extract_distilbert.py: student layer i ← teacher layer init[i]
    temperature: float = 2.0
    alpha_ce: float = 5.0  # the README's train.py command: --alpha_ce 5.0 --alpha_mlm 2.0 --alpha_cos 1.0
    alpha_mlm: float = 2.0
    alpha_cos: float = 1.0


# %% [init] The student from the teacher's layers
LAYER_NAMES = {  # DistilBERT TransformerBlock ← BertLayer
    "attention.q_lin": "attention.self.query",
    "attention.k_lin": "attention.self.key",
    "attention.v_lin": "attention.self.value",
    "attention.out_lin": "attention.output.dense",
    "sa_layer_norm": "attention.output.LayerNorm",
    "ffn.lin1": "intermediate.dense",
    "ffn.lin2": "output.dense",
    "output_layer_norm": "output.LayerNorm",
}


def student_from_teacher(teacher_state: dict, c: Config) -> dict:
    """A DistilBertForMaskedLM state dict from a BertForMaskedLM one: word and position embeddings (BERT's
    token-type embeddings and pooler have no place in the student), teacher layers init_layers, and the MLM head."""
    t = lambda name: teacher_state[name].clone()
    state = {
        "distilbert.embeddings.word_embeddings.weight": t("bert.embeddings.word_embeddings.weight"),
        "distilbert.embeddings.position_embeddings.weight": t("bert.embeddings.position_embeddings.weight"),
        "distilbert.embeddings.LayerNorm.weight": t("bert.embeddings.LayerNorm.weight"),
        "distilbert.embeddings.LayerNorm.bias": t("bert.embeddings.LayerNorm.bias"),
    }
    for i, j in enumerate(c.init_layers):
        for ours, theirs in LAYER_NAMES.items():
            for p in ("weight", "bias"):
                state[f"distilbert.transformer.layer.{i}.{ours}.{p}"] = t(f"bert.encoder.layer.{j}.{theirs}.{p}")
    for p in ("weight", "bias"):  # the MLM head: transform, LayerNorm, projection onto the vocabulary
        state[f"vocab_transform.{p}"] = t(f"cls.predictions.transform.dense.{p}")
        state[f"vocab_layer_norm.{p}"] = t(f"cls.predictions.transform.LayerNorm.{p}")
    state["vocab_projector.weight"] = t("cls.predictions.decoder.weight")  # tied to the word embeddings
    state["vocab_projector.bias"] = t("cls.predictions.bias")
    return state


# %% [soft-targets] Softened distributions
def soft(logits, temperature):
    """softmax(z / T): T = 1 is the model's prediction; a larger T moves probability onto the runners-up."""
    z = logits / temperature
    z = z - z.max(dim=-1, keepdim=True).values  # stable: exp of numbers ≤ 0
    return z.exp() / z.exp().sum(dim=-1, keepdim=True)


# %% [loss] The three losses
def distillation_loss(s_logits, t_logits, s_hidden, t_hidden, labels, attention_mask, c: Config):
    """s_/t_logits: (batch, seq, vocab); s_/t_hidden: the last hidden states (batch, seq, 768); labels: the
    masked words' ids, −100 elsewhere. Returns (L, L_ce, L_mlm, L_cos)."""
    keep = attention_mask.bool()  # every real token, not only the masked ones
    T = c.temperature
    p_t = soft(t_logits[keep], T)  # (tokens, vocab), the soft targets
    log_p_s = F.log_softmax(s_logits[keep] / T, dim=-1)
    l_ce = (p_t * (p_t.clamp_min(1e-30).log() - log_p_s)).sum(-1).mean() * T**2  # T²·KL(p_t ‖ p_s), mean per token

    scored = labels != -100  # the hard label: −log p_student(true word) at T = 1
    l_mlm = -F.log_softmax(s_logits[scored], dim=-1).gather(-1, labels[scored][:, None]).mean()

    cos = F.cosine_similarity(s_hidden[keep], t_hidden[keep], dim=-1)  # align the directions of the last states
    l_cos = (1 - cos).mean()
    return c.alpha_ce * l_ce + c.alpha_mlm * l_mlm + c.alpha_cos * l_cos, l_ce, l_mlm, l_cos
`;export{e as default};
