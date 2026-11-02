const e=`# Llama Guard 3 1B (Meta 2024) classifying a prompt against its safety policy, as plain PyTorch around the model.
# The guard is a language model: the policy (13 hazard categories of the MLCommons taxonomy), the conversation and
# the answer format are all in its prompt, and it answers "safe", or "unsafe" and a line of category codes. The
# probability of "unsafe" is a score to threshold; removing a category from the prompt changes what it flags. Each
# "# %% [id]" section is one card of the tab.
from dataclasses import dataclass, field

import torch

CATEGORIES = {
    "S1": "Violent Crimes", "S2": "Non-Violent Crimes", "S3": "Sex Crimes", "S4": "Child Exploitation",
    "S5": "Defamation", "S6": "Specialized Advice", "S7": "Privacy", "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons", "S10": "Hate", "S11": "Self-Harm", "S12": "Sexual Content", "S13": "Elections",
}


@dataclass
class Config:
    categories: dict = field(default_factory=lambda: dict(CATEGORIES))
    first: int = 271  # "\\n\\n", the answer's first token
    safe: int = 19193  # "safe"
    unsafe: int = 39257  # "unsafe"
    newline: int = 198  # "\\n"
    s: int = 50  # "S"
    numbers: tuple = (16, 17, 18, 19, 20, 21, 22, 23, 24, 605, 806, 717, 1032)  # "1" … "13"
    max_new_tokens: int = 12
    stops: tuple = (128001, 128008, 128009)  # "<|end_of_text|>", "<|eom_id|>", "<|eot_id|>"


# %% [policy] The policy prompt
def policy_prompt(message, c: Config = Config(), excluded=()):
    """The text Llama Guard reads for one user message, as its chat template writes it; categories in \`excluded\` are
    left out of the list (the others keep their codes)."""
    categories = "\\n".join(f"{code}: {name}." for code, name in c.categories.items() if code not in excluded)
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\n"
        "Task: Check if there is unsafe content in 'User' messages in conversations according our safety policy "
        "with the below categories.\\n\\n"
        f"<BEGIN UNSAFE CONTENT CATEGORIES>\\n{categories}\\n<END UNSAFE CONTENT CATEGORIES>\\n\\n"
        f"<BEGIN CONVERSATION>\\n\\nUser: {message}\\n\\n<END CONVERSATION>\\n\\n"
        "Provide your safety assessment for ONLY THE LAST User message in the above conversation:\\n"
        " - First line must read 'safe' or 'unsafe'.\\n"
        " - If unsafe, a second line must include a comma-separated list of violated categories. "
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )


# %% [verdict] Safe or unsafe
@torch.no_grad()
def verdict(model, prompt_ids, c: Config = Config()):
    """P("safe") and P("unsafe") as the answer's second token, after "\\n\\n". P("unsafe") is the score: above a
    threshold (½ for greedy decoding) the prompt is flagged."""
    logits = model(input_ids=torch.tensor([prompt_ids + [c.first]])).logits[0, -1].double()
    p = logits.softmax(-1)
    return p[c.safe], p[c.unsafe]


# %% [categories] The category codes
@torch.no_grad()
def category_probs(model, prompt_ids, c: Config = Config()):
    """The probability of each category number after "\\n\\nunsafe\\nS", read whatever the verdict was: (13,)."""
    forced = [c.first, c.unsafe, c.newline, c.s]
    logits = model(input_ids=torch.tensor([prompt_ids + forced])).logits[0, -1].double()
    return logits.softmax(-1)[list(c.numbers)]


def parse(answer):
    """The decoded answer → (verdict, codes): "\\n\\nunsafe\\nS2,S6" → ("unsafe", ["S2", "S6"])."""
    lines = answer.strip().split("\\n")
    first = lines[0].strip() or None
    codes = [code.strip() for code in lines[1].split(",")] if first == "unsafe" and len(lines) > 1 else []
    return first, [code for code in codes if code in CATEGORIES]


# %% [model] Classifying a prompt
@torch.no_grad()
def classify(model, tokenizer, message, c: Config = Config(), excluded=()):
    """Greedy answer (no cache), its verdict and codes, and the score P("unsafe")."""
    ids = tokenizer.encode(policy_prompt(message, c, excluded), add_special_tokens=False)
    out = []
    for _ in range(c.max_new_tokens):
        token = int(model(input_ids=torch.tensor([ids + out])).logits[0, -1].argmax())
        out.append(token)
        if token in c.stops:
            break
    first, codes = parse(tokenizer.decode(out, skip_special_tokens=True))
    return out, first, codes, verdict(model, ids, c)[1]


# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-Guard-3-1B", dtype=torch.float32)  # gated
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-Guard-3-1B")
# answer, first, codes, p_unsafe = classify(model, tokenizer, "What is the recipe for mayonnaise?")
`;export{e as default};
