const e=`# ReAct (Yao et al. 2023) with Qwen3-1.7B on questions about SQuAD's Wikipedia paragraphs, as plain PyTorch around
# Hugging Face models. The model writes a Thought and an Action as plain text; the loop stops it where it would write
# the Observation itself, runs the action's tool, pastes the tool's real output in as the Observation and lets the
# model continue, until it calls finish[...] or runs out of turns. The tools are local and deterministic: a search
# over the Retrieval Explainer's 2,067 SQuAD paragraphs (BAAI/bge-small-en-v1.5 vectors, exact search, as on the RAG
# tab) and a calculator for arithmetic. The whole run is one assistant message that grows turn by turn, decoded
# greedily with a key-value cache. The models are passed in (transformers' AutoModel classes); this file is what ReAct
# adds around them. Each "# %% [id]" section is one card of the tab.
import ast
import operator
import re
from dataclasses import dataclass

import torch


@dataclass
class Config:
    max_turns: int = 8  # Thought/Action/Observation rounds before the loop gives up
    max_new_tokens: int = 256  # per turn
    stop: str = 'Observation:'  # where the model would start writing the tool's output itself
    top: int = 8  # next-token candidates recorded per step
    search_k: int = 2  # paragraphs a search returns
    query_instruction: str = 'Represent this sentence for searching relevant passages: '  # bge's query prefix
    max_length: int = 512  # tokens the encoder reads


SYSTEM = """Answer the question by interleaving Thought, Action and Observation steps. A Thought reasons about \\
what is known so far and what is still missing. An Action is one of:
search[query]: searches Wikipedia and returns the two most relevant paragraphs.
calculate[expression]: evaluates an arithmetic expression, such as 2016 - 1998.
finish[answer]: gives the final answer, a short phrase or a number, and ends.
After each Action you will get an Observation with its result. Write one Thought and one Action at a time:
Thought: ...
Action: ..."""


# %% [prompt] The ReAct prompt
def messages(question):
    """The system turn describes the tools and the format; the user turn is the question. No worked examples."""
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': f'Question: {question}'}]


def prompt_ids(tokenizer, question):
    """The chat template's tokens, ending with the assistant header. Thinking is off: Qwen3's template then writes an
    empty <think></think> block, and the ReAct Thoughts do the reasoning in plain view."""
    return tokenizer.apply_chat_template(messages(question), add_generation_prompt=True, tokenize=True,
                                         enable_thinking=False, return_tensors='pt')[0]


# %% [generate] One turn: greedy tokens until the stop string
@torch.no_grad()
def turn(lm, tokenizer, ids, eos_ids, cfg=Config()):
    """Continue the message from \`ids\`: append the most likely token, one at a time, until the text written this turn
    contains the stop string, an end token comes, or max_new_tokens. The first step reads the whole message; after
    that a key-value cache holds every earlier token, so each step runs only the newest one (transformers' generate
    with stop_strings does the same). Returns (tokens, per-step (top ids, top probabilities), why it stopped, how
    many tokens to keep): the tokens that overlap the stop string are dropped, so the model never reads its own
    guess of the observation."""
    out = lm(input_ids=ids[None].to(lm.device), use_cache=True)
    tokens, tops, reason = [], [], 'length'
    for _ in range(cfg.max_new_tokens):
        probs = out.logits[0, -1].float().softmax(-1)
        p, i = probs.topk(cfg.top)
        tops.append((i.cpu(), p.cpu()))
        token = int(i[0])
        tokens.append(token)
        if token in eos_ids:
            reason = 'eos'
            break
        if cfg.stop in tokenizer.decode(tokens):
            reason = 'stop'
            break
        out = lm(input_ids=torch.tensor([[token]], device=lm.device), past_key_values=out.past_key_values,
                 use_cache=True)
    keep = len(tokens) - 1 if reason == 'eos' else len(tokens)
    if reason == 'stop':  # the longest prefix whose text ends before the stop string begins
        start = tokenizer.decode(tokens).index(cfg.stop)
        keep = max(n for n in range(len(tokens) + 1) if len(tokenizer.decode(tokens[:n])) <= start)
    return tokens, tops, reason, keep


# %% [parse] Reading the action
ACTION = re.compile(r'Action:\\s*(\\w+)\\s*\\[(.*)\\]', re.S)


def parse(text):
    """The last 'Action: tool[argument]' in a turn's text (the argument may hold brackets) and the Thought before
    it. Returns (thought, tool, argument), with tool None when no action can be read."""
    thought = re.search(r'Thought:\\s*(.*?)(?=\\n\\s*Action:|$)', text, re.S)
    thought = thought[1].strip() if thought else ''
    found = list(ACTION.finditer(text))
    if not found:
        return thought, None, text.strip()
    action = found[-1]
    return thought, action[1].lower(), action[2].strip()


# %% [search] The search tool
@torch.no_grad()
def embed(encoder, tokenizer, query, cfg=Config()):
    """bge-small's query vector: bge's instruction, the last layer's [CLS] state scaled to length 1, rounded to
    float16 like the paragraph vectors (the RAG tab's embed)."""
    ids = tokenizer(cfg.query_instruction + query, truncation=True, max_length=cfg.max_length, return_tensors='pt')
    cls = encoder(**ids).last_hidden_state[0, 0]
    return torch.nn.functional.normalize(cls, dim=-1).half().float()


def search(query, encoder, tokenizer, vectors, docs, cfg=Config()):
    """Exact search: squared distance from the query vector to all 2,067 paragraph vectors, the search_k nearest.
    Returns their ids, distances and the observation text: each paragraph under its article's title."""
    q = embed(encoder, tokenizer, query, cfg)
    d = ((vectors.double() - q.double()) ** 2).sum(-1)
    order = d.argsort(stable=True)[: cfg.search_k]
    text = ' '.join(f"[{r + 1}] {docs[i]['title'].replace('_', ' ')}: {docs[i]['text']}"
                    for r, i in enumerate(order.tolist()))
    return order, d[order], text


# %% [calculate] The calculator
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
       ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculate(expression):
    """Numbers, + − × ÷ // % ** and brackets, parsed by Python's own parser and walked node by node, so nothing but
    arithmetic can run. Whole results print without '.0'. Errors come back as the observation text."""
    def value(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](value(node.left), value(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](value(node.operand))
        raise ValueError(f'not arithmetic: {ast.unparse(node)}')

    try:
        result = value(ast.parse(expression.replace(',', '').replace('^', '**'), mode='eval').body)
    except Exception as error:  # noqa: BLE001 - the error is the observation
        return f'Error: {error}'
    return str(int(result)) if float(result).is_integer() and abs(result) < 1e15 else f'{result:.6g}'


# %% [loop] The loop: thought, action, observation
def observe(tool, argument, tools):
    """Run one action. finish ends the loop; an unknown tool or an unreadable action is reported back to the model,
    as the ReAct paper's environment does, and the loop goes on."""
    if tool == 'search':
        return tools['search'](argument)[2]
    if tool == 'calculate':
        return calculate(argument)
    if tool is None:
        return 'Invalid action. Write Action: search[query], calculate[expression] or finish[answer].'
    return f'Invalid action: {tool}. The tools are search, calculate and finish.'


def react(lm, tokenizer, tools, question, cfg=Config()):
    """Prompt, then turns until finish[...] or max_turns. After every turn the tokens the model wrote (minus its
    attempt at an Observation) stay in the message and the real observation's tokens are appended after them, so the
    next turn reads everything so far. Returns the turns and the answer (None if the loop never finished)."""
    eos = set(lm.generation_config.eos_token_id)
    ids, turns = prompt_ids(tokenizer, question), []
    for _ in range(cfg.max_turns):
        tokens, tops, reason, keep = turn(lm, tokenizer, ids, eos, cfg)
        thought, tool, argument = parse(tokenizer.decode(tokens[:keep]))
        turns.append({'tokens': tokens, 'tops': tops, 'reason': reason, 'keep': keep, 'thought': thought,
                      'tool': tool, 'argument': argument})
        if tool == 'finish':
            return turns, argument
        observation = observe(tool, argument, tools)
        appended = tokenizer(f'Observation: {observation}\\n', add_special_tokens=False, return_tensors='pt').input_ids[0]
        turns[-1]['observation'] = observation
        ids = torch.cat([ids, torch.tensor(tokens[:keep], dtype=torch.long), appended])
    return turns, None


# %% [direct] Answering without tools
@torch.no_grad()
def direct(lm, tokenizer, question, max_new_tokens=64):
    """The same model, no tools, no loop: the question alone, answered greedily with thinking off (the baseline)."""
    chat = [{'role': 'user', 'content': f'{question} Answer with a short phrase or a number.'}]
    ids = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, enable_thinking=False,
                                        return_tensors='pt')
    out = lm.generate(ids.to(lm.device), attention_mask=torch.ones_like(ids).to(lm.device), do_sample=False,
                      max_new_tokens=max_new_tokens, temperature=None, top_p=None, top_k=None)
    return ids[0], out[0, ids.shape[1]:].cpu()


# lm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float32)
# encoder = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5"); vectors: static/traces/retrieval/hnsw/vectors.f16
# tools = {"search": lambda q: search(q, encoder, encoder_tokenizer, vectors, docs)}
# turns, answer = react(lm, tokenizer, tools, "How many years apart were Nikola Tesla and Martin Luther born?")
`;export{e as default};
