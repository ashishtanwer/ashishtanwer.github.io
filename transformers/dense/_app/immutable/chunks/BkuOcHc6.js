const n=`# Tool calling with Qwen3-1.7B, as plain PyTorch and Python.
# The tools' JSON schemas go into the prompt: Qwen3's chat template writes them into the system message between
# <tools> tags and tells the model to answer a call as JSON between <tool_call> tokens. The model then either
# writes text (the answer) or a <tool_call> block; the caller parses the block, runs the tool itself, and feeds
# the result back as a tool message, which the template wraps in <tool_response>. The loop ends on a turn with no
# call. Decoding is greedy. Each "# %% [id]" section is one card of the tab.
import ast
import json
import re
from dataclasses import dataclass
from decimal import Decimal, getcontext

import torch


@dataclass
class Config:
    max_new_tokens: int = 256  # per assistant turn
    max_turns: int = 6  # assistant turns before the loop gives up
    stops: tuple = (151645, 151643)  # "<|im_end|>", "<|endoftext|>"
    enable_thinking: bool = False  # Qwen3's non-thinking mode: the template writes an empty <think></think>


# %% [template] The prompt the chat template builds
TOOLS_HEADER = (
    "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\n"
    "You are provided with function signatures within <tools></tools> XML tags:\\n<tools>"
)
TOOLS_FOOTER = (
    "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within "
    '<tool_call></tool_call> XML tags:\\n<tool_call>\\n{"name": <function-name>, "arguments": <args-json-object>}\\n'
    "</tool_call><|im_end|>\\n"
)


def is_tool_response(message):
    content = message.get("content")
    return isinstance(content, str) and content.startswith("<tool_response>") and content.endswith("</tool_response>")


def render(messages, tools=None, add_generation_prompt=True, enable_thinking=False):
    """Qwen3's chat template (its Jinja source, in Python): a system message with the tools' JSON schemas, then
    every message between <|im_start|>role and <|im_end|>. Tool results become one user turn of <tool_response>
    blocks. An assistant turn keeps its <think> block only if it comes after the last real user message and is
    the last message or has reasoning, so earlier turns' empty <think></think> drop out of the prompt."""
    out = ""
    first_system = messages and messages[0]["role"] == "system"
    if tools:
        out += "<|im_start|>system\\n" + (messages[0]["content"] + "\\n\\n" if first_system else "") + TOOLS_HEADER
        out += "".join("\\n" + json.dumps(tool, ensure_ascii=False) for tool in tools) + TOOLS_FOOTER
    elif first_system:
        out += "<|im_start|>system\\n" + messages[0]["content"] + "<|im_end|>\\n"
    last_query = next((i for i in range(len(messages) - 1, -1, -1)
                       if messages[i]["role"] == "user" and isinstance(messages[i]["content"], str)
                       and not is_tool_response(messages[i])), len(messages) - 1)
    for i, m in enumerate(messages):
        content = m["content"] if isinstance(m.get("content"), str) else ""
        last = i == len(messages) - 1
        if m["role"] == "user" or (m["role"] == "system" and i > 0):
            out += f"<|im_start|>{m['role']}\\n{content}<|im_end|>\\n"
        elif m["role"] == "assistant":
            reasoning = m.get("reasoning_content")
            if not isinstance(reasoning, str):
                reasoning = ""
                if "</think>" in content:
                    reasoning = content.split("</think>")[0].rstrip("\\n").split("<think>")[-1].lstrip("\\n")
                    content = content.split("</think>")[-1].lstrip("\\n")
            if i > last_query and (last or reasoning):
                out += "<|im_start|>assistant\\n<think>\\n" + reasoning.strip("\\n") + "\\n</think>\\n\\n" + content.lstrip("\\n")
            else:
                out += "<|im_start|>assistant\\n" + content
            for k, call in enumerate(m.get("tool_calls") or []):
                call = call.get("function", call)
                arguments = call["arguments"] if isinstance(call["arguments"], str) else json.dumps(
                    call["arguments"], ensure_ascii=False)
                out += ("\\n" if (k == 0 and content) or k > 0 else "")
                out += '<tool_call>\\n{"name": "' + call["name"] + '", "arguments": ' + arguments + "}\\n</tool_call>"
            out += "<|im_end|>\\n"
        elif m["role"] == "tool":
            if i == 0 or messages[i - 1]["role"] != "tool":
                out += "<|im_start|>user"
            out += "\\n<tool_response>\\n" + content + "\\n</tool_response>"
            if last or messages[i + 1]["role"] != "tool":
                out += "<|im_end|>\\n"
    if add_generation_prompt:
        out += "<|im_start|>assistant\\n" + ("" if enable_thinking else "<think>\\n\\n</think>\\n\\n")
    return out


# %% [output] The next token
def next_token(logits):
    """logits (V,) → (the greedy token, the probability of every token). Greedy takes the most likely token; its
    probability is how sure the model was, e.g. of starting a <tool_call> rather than writing an answer."""
    probs = logits.double().softmax(-1)
    return int(logits.argmax()), probs


@torch.no_grad()
def assistant_turn(model, prompt_ids, c: Config = Config()):
    """One assistant turn, token by token, without a cache: the generated ids (the stop token included) and each
    token's probability."""
    ids, out, p = list(prompt_ids), [], []
    for _ in range(c.max_new_tokens):
        token, probs = next_token(model(input_ids=torch.tensor([ids])).logits[0, -1])
        ids.append(token)
        out.append(token)
        p.append(float(probs[token]))
        if token in c.stops:
            break
    return out, p


# %% [parse] Parsing the tool calls
CALL = re.compile(r"<tool_call>\\n(.+)?\\n</tool_call>")


def parse_assistant(text):
    """The assistant turn's text (special tokens kept, the stop token dropped) → a message: every
    <tool_call>…</tool_call> block read as JSON {"name": …, "arguments": {…}}, and the text before the first call as
    its content. Qwen's function-calling guide parses it this way; a block that is not JSON is not a call."""
    calls, offset = [], 0
    for k, match in enumerate(CALL.finditer(text)):
        if k == 0:
            offset = match.start()
        try:
            function = json.loads(match.group(1))
            if isinstance(function["arguments"], str):
                function["arguments"] = json.loads(function["arguments"])
            calls.append({"type": "function", "function": function})
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    if calls:
        content = text[:offset] if offset > 0 and text[:offset].strip() else ""
        return {"role": "assistant", "content": content, "tool_calls": calls}
    return {"role": "assistant", "content": text}


# %% [tools] The tools
getcontext().prec = 28
OPERATORS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
             ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a**b, ast.Mod: lambda a, b: a % b}


def calculator(expression):
    """Exact decimal arithmetic on + − × ÷ **, % and parentheses (28 significant digits), nothing else: the expression
    is parsed, never run as Python. Returns the result as text, or an error message the model can read."""
    def value(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return Decimal(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -value(node.operand) if isinstance(node.op, ast.USub) else value(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](value(node.left), value(node.right))
        raise ValueError(f"unsupported: {ast.dump(node)[:40]}")

    try:
        result = value(ast.parse(expression.replace("^", "**").replace(",", ""), mode="eval").body)
    except (SyntaxError, ValueError, ArithmeticError) as error:
        return f"Error: {error}"
    return str(result.to_integral_value()) if result == result.to_integral_value() else str(result.normalize())


def lookup_element(element, table):
    """One row of a periodic table (PubChem's), found by name or symbol, as JSON text."""
    key = element.strip().lower()
    for row in table:
        if key in (row["Name"].lower(), row["Symbol"].lower()):
            return json.dumps({"atomic_number": int(row["AtomicNumber"]), "symbol": row["Symbol"], "name": row["Name"],
                               "atomic_mass": float(row["AtomicMass"]), "standard_state": row["StandardState"],
                               "year_discovered": row["YearDiscovered"]})
    return json.dumps({"error": f"no element named {element!r}"})


def run_tool(call, table):
    """Run one parsed call; the tool message carries its result back."""
    name, arguments = call["function"]["name"], call["function"]["arguments"]
    if name == "calculator":
        result = calculator(str(arguments.get("expression", "")))
    elif name == "lookup_element":
        result = lookup_element(str(arguments.get("element", "")), table)
    else:
        result = f"Error: no tool named {name!r}"
    return {"role": "tool", "content": result}


# %% [model] Tool calling
def tool_loop(model, tokenizer, messages, tools, table, c: Config = Config()):
    """Prompt, turn, parse, run, until a turn makes no call. Every turn re-renders the whole conversation through
    the template, as an agent framework does, so the prompt is exactly what the template builds."""
    messages = list(messages)
    for _ in range(c.max_turns):
        prompt = tokenizer(render(messages, tools, enable_thinking=c.enable_thinking), add_special_tokens=False)
        ids, _ = assistant_turn(model, prompt["input_ids"], c)
        text = tokenizer.decode([i for i in ids if i not in c.stops])
        message = parse_assistant(text)
        messages.append(message)
        if "tool_calls" not in message:
            break
        messages += [run_tool(call, table) for call in message["tool_calls"]]
    return messages


# TOOLS = [{"type": "function", "function": {"name": "calculator", "description": …, "parameters": {…}}}, …]
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float32)
# messages = tool_loop(model, tokenizer, [{"role": "user", "content": question}], TOOLS, periodic_table)
`;export{n as default};
