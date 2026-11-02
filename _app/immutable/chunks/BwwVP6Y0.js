const e=`# CodeAct (Wang et al. 2024) with Qwen3-1.7B, as plain PyTorch around a Hugging Face model: the agent's actions are
# Python code blocks, run in a sandboxed interpreter like a notebook's, whose output (what the code prints and the
# value of a last bare expression, or its traceback) comes back as the next message, until the model replies with an
# answer instead of code. The same tasks also run with JSON tool calls, the chat template's own format: the model names one function and its arguments per call, each call runs on its own and
# its return value comes back as a tool message. Both agents get the same three functions (is_prime, weekday,
# days_between; scripts/agents/sandbox_repl.py), the same model and greedy decoding; the code agent can also use the
# whole Python standard library, loops and variables. The conversation is re-rendered by the chat template every turn,
# as a chat server does. Each "# %% [id]" section is one card of the tab.
import json
import os
import re
import select
import subprocess
import tempfile
import time
from dataclasses import dataclass

import torch


@dataclass
class Config:
    max_turns: int = 8  # replies before the loop gives up
    max_new_tokens: int = 512  # per reply
    top: int = 8  # next-token candidates recorded per step
    time_limit: float = 10.0  # seconds one action may run before the interpreter is killed
    max_output: int = 2000  # characters of an observation the model is shown


FUNCTIONS = """is_prime(n: int) -> bool: whether the whole number n is prime.
weekday(date: str) -> str: the day of the week of a date written YYYY-MM-DD, such as weekday("2016-02-07").
days_between(start: str, end: str) -> int: the number of days from date start to date end, both YYYY-MM-DD."""
ANSWER = 'When you know the answer, reply with "Answer: " followed by the answer, a number or a short phrase'
CODE_SYSTEM = f"""You solve tasks by writing Python code. To act, reply with one Python code block:
\`\`\`python
...
\`\`\`
It runs in a Python interpreter that keeps its variables between your actions, like a Jupyter notebook: you see what \\
the code prints and the value of its last line. Besides the standard library, these functions are already defined:
{FUNCTIONS}
After each code block you get an Observation with its output, or its error. {ANSWER}, and no code."""
JSON_SYSTEM = f'You solve tasks by calling the functions you are given. {ANSWER}, and no function call.'


def schema(name, description, **parameters):
    """One function in the JSON Schema form the chat template writes into the prompt."""
    return {'type': 'function', 'function': {'name': name, 'description': description, 'parameters': {
        'type': 'object', 'properties': {k: {'type': t, 'description': d} for k, (t, d) in parameters.items()},
        'required': list(parameters)}}}


SCHEMAS = [
    schema('is_prime', 'Whether the whole number n is prime.', n=('integer', 'A whole number')),
    schema('weekday', 'The day of the week of a date.', date=('string', 'A date written YYYY-MM-DD')),
    schema('days_between', 'The number of days from date start to date end.',
           start=('string', 'A date written YYYY-MM-DD'), end=('string', 'A date written YYYY-MM-DD')),
]


# %% [prompt] Two prompts for one task
def code_messages(task):
    return [{'role': 'system', 'content': CODE_SYSTEM}, {'role': 'user', 'content': task}]


def json_messages(task):
    return [{'role': 'system', 'content': JSON_SYSTEM}, {'role': 'user', 'content': task}]


def chat_ids(tokenizer, messages, tools=None):
    """The whole conversation so far through Qwen3's chat template, thinking off, ending with the assistant header.
    With \`tools\` the template adds them to the system turn as JSON Schemas, with its instructions for <tool_call>."""
    return tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=True,
                                         enable_thinking=False, return_tensors='pt')[0]


# %% [generate] One reply: greedy tokens until the end of the message
@torch.no_grad()
def reply(lm, ids, eos_ids, cfg=Config()):
    """Append the most likely token until an end-of-message token or max_new_tokens; after the first step a key-value
    cache holds the earlier tokens. Returns (tokens, per-step (top ids, top probabilities), why it stopped)."""
    out = lm(input_ids=ids[None].to(lm.device), use_cache=True)
    tokens, tops = [], []
    for _ in range(cfg.max_new_tokens):
        p, i = out.logits[0, -1].float().softmax(-1).topk(cfg.top)
        tops.append((i.cpu(), p.cpu()))
        tokens.append(int(i[0]))
        if tokens[-1] in eos_ids:
            return tokens, tops, 'eos'
        out = lm(input_ids=torch.tensor([[tokens[-1]]], device=lm.device), past_key_values=out.past_key_values,
                 use_cache=True)
    return tokens, tops, 'length'


# %% [parse] Reading the action
CODE = re.compile(r'\`\`\`(?:python|py)?[^\\n]*\\n(.*?)\`\`\`', re.S)
CALL = re.compile(r'<tool_call>\\s*(.*?)\\s*</tool_call>', re.S)


def answer_of(text):
    """The text after the last "Answer:", or the whole reply when the model answered without the marker."""
    found = re.findall(r'Answer:\\s*(.+)', text)
    return (found[-1] if found else text).strip()


def parse_code(text):
    """(code, answer): the reply's code blocks joined in order, or None and the answer when it has none."""
    blocks = CODE.findall(text)
    return ('\\n'.join(b.rstrip() for b in blocks), None) if blocks else (None, answer_of(text))


def parse_calls(text):
    """(calls, answer): every <tool_call> as {name, arguments} (or {error} when its JSON does not parse), or no calls
    and the answer."""
    calls = []
    for body in CALL.findall(text):
        try:
            call = json.loads(body)
            calls.append({'name': call['name'], 'arguments': call.get('arguments', {})})
        except (ValueError, KeyError, TypeError) as error:
            calls.append({'error': f'could not read the call: {error}', 'raw': body})
    return (calls, None) if calls else ([], answer_of(text))


# %% [sandbox] The sandbox: a Python with no network and a time limit
class Sandbox:
    """One persistent interpreter (scripts/agents/sandbox_repl.py) started with macOS's sandbox-exec: every network
    operation and every file write outside a fresh empty folder is denied, and \`python -I -S -B\` leaves only the
    standard library and the three functions importable. Each action is sent as a JSON line; the interpreter runs it
    in the namespace earlier actions left behind and answers with what it printed (then the value of a last bare
    expression, as a notebook cell shows it) and its traceback. An action still running after time_limit seconds gets
    the process killed and a fresh one started, so its variables are lost."""

    def __init__(self, repl, python, cfg=Config()):
        self.repl, self.python, self.cfg = repl, python, cfg
        self.start()

    def start(self):
        self.folder = os.path.realpath(tempfile.mkdtemp(prefix='codeact-'))
        profile = ('(version 1)(allow default)(deny network*)(deny file-write*)'
                   f'(allow file-write* (subpath "{self.folder}"))(allow file-write* (literal "/dev/null"))')
        self.process = subprocess.Popen(
            ['/usr/bin/sandbox-exec', '-p', profile, self.python, '-I', '-S', '-B', self.repl], cwd=self.folder,
            env={'PATH': '/usr/bin:/bin'}, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True)

    def run(self, code):
        """{stdout, stderr, error, seconds, timedOut} of one action."""
        began = time.perf_counter()
        self.process.stdin.write(json.dumps({'code': code}) + '\\n')
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], self.cfg.time_limit)
        if not ready:
            self.close()
            self.start()
            error = (f'TimeoutError: the code ran for more than {self.cfg.time_limit:g} seconds and was stopped. The '
                     'interpreter was restarted, so variables from earlier actions are gone.')
            return {'stdout': '', 'stderr': '', 'error': error, 'seconds': self.cfg.time_limit, 'timedOut': True}
        result = json.loads(self.process.stdout.readline())
        return {**result, 'seconds': time.perf_counter() - began, 'timedOut': False}

    def close(self):
        self.process.kill()
        self.process.wait()


def observation(result, cfg=Config()):
    """What the model is shown: the output (prints, then a last expression's value), then the traceback; '(no output)'
    when there is nothing."""
    text = (result['stdout'] + result['stderr'] + (result['error'] or '')).strip()
    if len(text) > cfg.max_output:
        text = f'{text[: cfg.max_output]}\\n… ({len(text) - cfg.max_output} more characters)'
    return text or '(no output)'


# %% [codeact] The CodeAct loop
def codeact(lm, tokenizer, task, sandbox, cfg=Config()):
    """Reply, run its code, show the output as the next user message ("Observation: …"), until a reply has no code
    or max_turns. Returns the turns and the answer (None if the loop never ended)."""
    eos = set(lm.generation_config.eos_token_id)
    messages, turns = code_messages(task), []
    for _ in range(cfg.max_turns):
        ids = chat_ids(tokenizer, messages)
        tokens, tops, reason = reply(lm, ids, eos, cfg)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        code, answer = parse_code(text)
        turns.append({'prompt': ids, 'tokens': tokens, 'tops': tops, 'reason': reason, 'text': text, 'code': code})
        messages.append({'role': 'assistant', 'content': text})
        if code is None:
            return turns, answer
        result = sandbox.run(code)
        turns[-1].update(result=result, observation=observation(result, cfg))
        messages.append({'role': 'user', 'content': f'Observation:\\n{turns[-1]["observation"]}'})
    return turns, None


# %% [json] The same tasks with JSON tool calls
def call(functions, name, arguments):
    """Run one parsed call: the function's return value as JSON, or the error, as the tool message's text."""
    if name not in functions:
        return f'Error: there is no function {name}'
    try:
        return json.dumps(functions[name](**arguments))
    except Exception as error:  # noqa: BLE001 - the error goes back to the model
        return f'Error: {type(error).__name__}: {error}'


def tool_calling(lm, tokenizer, task, functions, cfg=Config()):
    """Reply; run each <tool_call> in it and answer each with a tool message, until a reply has no call or
    max_turns. The assistant turn goes back into the conversation as structured tool_calls, which the template
    writes out again as JSON."""
    eos = set(lm.generation_config.eos_token_id)
    messages, turns = json_messages(task), []
    for _ in range(cfg.max_turns):
        ids = chat_ids(tokenizer, messages, SCHEMAS)
        tokens, tops, reason = reply(lm, ids, eos, cfg)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        calls, answer = parse_calls(text)
        turns.append({'prompt': ids, 'tokens': tokens, 'tops': tops, 'reason': reason, 'text': text, 'calls': calls})
        if not calls:
            messages.append({'role': 'assistant', 'content': text})
            return turns, answer
        good = [c for c in calls if 'error' not in c]
        messages.append({'role': 'assistant', 'content': CALL.sub('', text).strip(), 'tool_calls': [
            {'type': 'function', 'function': {'name': c['name'], 'arguments': c['arguments']}} for c in good]})
        results = [f"Error: {c['error']}" if 'error' in c else call(functions, c['name'], c['arguments'])
                   for c in calls]
        turns[-1]['results'] = results
        messages += [{'role': 'tool', 'content': r} for r in results]
    return turns, None


# lm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float32)
# sandbox = Sandbox("scripts/agents/sandbox_repl.py", "/usr/bin/python3")
# turns, answer = codeact(lm, tokenizer, "How many prime numbers are there between 1,000 and 1,100?", sandbox)
`;export{e as default};
