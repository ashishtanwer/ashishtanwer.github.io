const e=`# pass@k (Chen et al. 2021, the HumanEval paper) on Qwen2.5-Coder-0.5B-Instruct, as plain PyTorch.
# Sample n programs per problem, run each against the problem's unit tests, count the c that pass, and estimate
# pass@k, the chance that at least one of k samples passes, with the unbiased estimator 1 − C(n−c, k) / C(n, k).
# Every program draws its tokens from its own seeded stream of uniforms, so a token reproduces from its logits and
# the seed. Each "# %% [id]" section is one card of the tab.
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Config:
    samples: int = 20  # n programs per problem
    temperature: float = 0.8  # the HumanEval paper's best for large k (0.2 for pass@1)
    top_k: int = 50
    top_p: float = 0.95
    max_new_tokens: int = 512
    stops: tuple = (151645, 151643)  # "<|im_end|>", "<|endoftext|>"
    timeout: float = 3.0  # seconds per program, HumanEval's default


# %% [sampling] Sampling a program's tokens
def sampling_probs(logits, temperature, top_k, top_p):
    """logits (..., V) → the distribution a token is drawn from: divide by the temperature, keep the top_k, then
    drop the least likely tokens whose total probability is at most 1 − top_p (transformers' warpers, in order)."""
    scores = logits / temperature
    kth = scores.topk(min(top_k, scores.shape[-1]), dim=-1).values[..., -1:]
    scores = scores.masked_fill(scores < kth, float("-inf"))
    ascending, order = scores.sort(dim=-1)
    drop = ascending.softmax(-1).cumsum(-1) <= 1 - top_p
    drop[..., -1:] = False
    return scores.masked_fill(drop.scatter(-1, order, drop), float("-inf")).softmax(-1)


def draw(probs, u):
    """Inverse CDF: tokens laid out on [0, 1) from the most likely down; the one whose slice holds u is drawn."""
    ordered, order = probs.sort(dim=-1, descending=True, stable=True)
    j = (ordered.cumsum(-1) <= u[..., None]).sum(-1)
    j = torch.minimum(j, (ordered > 0).sum(-1) - 1)
    return order.gather(-1, j[..., None])[..., 0]


def uniforms(seed, length):
    """Program j of HumanEval/t is seeded 1000·t + j, the same stream at every temperature."""
    return torch.rand(length, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


# %% [execute] Running a program against the tests
SANDBOX = ('(version 1)(allow default)(deny network*)(deny process-fork)(deny file-write*)'
           '(allow file-write* (subpath "{folder}") (literal "/dev/null"))')


def program(problem, code):
    """What the harness runs: the prompt's imports, the code (after the prompt if it does not define the function),
    the problem's tests and check(<entry point>)."""
    head = problem["prompt"].split("\\ndef ")[0]
    imports = "".join(line + "\\n" for line in head.splitlines() if line.startswith(("import ", "from ")))
    defines = re.search(rf"^def {re.escape(problem['entryPoint'])}\\(", code, re.M)
    body = code if defines else problem["prompt"] + code
    return f"{imports}{body}\\n\\n{problem['test']}\\n\\ncheck({problem['entryPoint']})\\n"


def execute(source, timeout=3.0):
    """Run a program in a fresh folder under macOS sandbox-exec (no network, no new processes, writes only in the
    folder), Python in isolated mode; 'passed', the exception it died of, or 'timed out'."""
    with tempfile.TemporaryDirectory(prefix="humaneval-") as folder:
        folder = os.path.realpath(folder)
        Path(folder, "program.py").write_text(source)
        try:
            done = subprocess.run(["sandbox-exec", "-p", SANDBOX.format(folder=folder), sys.executable, "-I", "-B",
                                   "program.py"], cwd=folder, capture_output=True, text=True, timeout=timeout,
                                  env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"})
        except subprocess.TimeoutExpired:
            return "timed out"
        if done.returncode == 0:
            return "passed"
        lines = [line for line in done.stderr.strip().splitlines() if line.strip()]
        m = re.match(r"^([A-Za-z_][\\w.]*)(?::|$)", lines[-1]) if lines else None
        return m[1].split(".")[-1] if m else f"exit {done.returncode}"


# %% [pass-at-k] The pass@k estimators
def pass_at_k(n, c, k):
    """Unbiased: of all C(n, k) ways to pick k of the n samples, the share that include a passing one."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def naive_pass_at_k(n, c, k):
    """1 − (1 − c/n)^k: k independent draws at the observed pass rate. Below pass@k on average."""
    return 1.0 - (1.0 - c / n) ** k


# %% [model] Sampling n programs and scoring them
@torch.no_grad()
def sample_program(model, prompt_ids, u, c: Config = Config(), greedy=False, decode=None):
    """One program, token by token (no cache): greedy takes the most likely token, sampling uses u[t] for token t.
    Stops at a stop token, after max_new_tokens, or (given decode(token) → its text) at the closing code fence."""
    ids, out, text = list(prompt_ids), [], ""
    for t in range(c.max_new_tokens):
        logits = model(input_ids=torch.tensor([ids])).logits[0, -1].double()
        token = int(logits.argmax()) if greedy else int(draw(sampling_probs(logits, c.temperature, c.top_k, c.top_p), u[t]))
        ids.append(token)
        out.append(token)
        text += decode(token) if decode else ""
        if token in c.stops or "\`\`\`" in text:
            break
    return out


def pass_rate(model, tokenizer, problem, prompt_ids, c: Config = Config(), ks=(1, 5, 10)):
    """n sampled programs for one problem, their test results and pass@k."""
    results = []
    for j in range(c.samples):
        u = uniforms(1000 * problem["task"] + j, c.max_new_tokens + 8)
        tokens = sample_program(model, prompt_ids, u, c, decode=lambda token: tokenizer.decode([token]))
        code = tokenizer.decode(tokens, skip_special_tokens=True).split("\`\`\`")[0]
        results.append(execute(program(problem, code), c.timeout))
    passed = sum(r == "passed" for r in results)
    return results, {k: pass_at_k(c.samples, passed, k) for k in ks}


# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct", dtype=torch.float32)
# prompt_ids = evalplus_chat_prompt(tokenizer, problem["prompt"])  # the user turn and "\`\`\`python\\n" begun
# results, estimates = pass_rate(model, tokenizer, problem, prompt_ids)
`;export{e as default};
