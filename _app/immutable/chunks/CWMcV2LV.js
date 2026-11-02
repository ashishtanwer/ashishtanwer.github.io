const e=`# The Model Context Protocol (MCP) with Qwen3-1.7B, as plain Python around the official \`mcp\` SDK and a Hugging Face
# model. A local server offers three tools over stdio (search over SQuAD's Wikipedia paragraphs, a calculator, the
# day of the week of a date); a client starts it, shakes hands (initialize), asks for its tools (tools/list), turns
# that list into the tool schemas Qwen3's chat template writes into the prompt, and sends every call the model makes
# to the server as a tools/call request, whose result goes back to the model as a tool message. Every message is one
# line of JSON-RPC 2.0 on the server's stdin or stdout. Each "# %% [id]" section is one card of the tab.
import datetime
import json
import re
from dataclasses import dataclass
from typing import Annotated

import torch
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field


@dataclass
class Config:
    max_turns: int = 6  # replies per question before the loop gives up
    max_new_tokens: int = 256  # per reply
    top: int = 8  # next-token candidates recorded per step


INSTRUCTIONS = 'Look things up in Wikipedia paragraphs and compute.'
ANSWER = 'When you know the answer, reply with "Answer: " followed by the answer, a short phrase or a number.'


# %% [server] The server: three Python functions become tools
def build_server(search_text, calculate_text):
    """An MCP server whose tools are plain functions. The SDK reads each one's name, docstring and annotated
    parameters into a tool with a JSON Schema for its arguments (tools/list), checks the arguments of every
    tools/call against it, runs the function and sends its return value back as text content. An exception becomes
    a result with isError: true and the exception's message as text."""
    server = MCPServer('squad-tools', version='1.0.0', instructions=INSTRUCTIONS)

    @server.tool(structured_output=False)
    def search(query: Annotated[str, Field(description='What to look for, in a few words')]) -> str:
        """Search Wikipedia and return the two most relevant paragraphs."""
        return search_text(query)

    @server.tool(structured_output=False)
    def calculate(expression: Annotated[str, Field(description='An arithmetic expression, such as 2016 - 1998')]) -> str:
        """Evaluate an arithmetic expression."""
        return calculate_text(expression)

    @server.tool(structured_output=False)
    def day_of_week(date: Annotated[str, Field(description='A date written YYYY-MM-DD')]) -> str:
        """Give the day of the week of a date."""
        try:
            return datetime.date.fromisoformat(date).strftime('%A')
        except ValueError:
            raise ToolError(f'{date!r} is not a date written YYYY-MM-DD') from None

    return server


# %% [schemas] From tools/list to the model's tool schemas
def to_schema(tool):
    """One tool of the tools/list result in the OpenAI-style function form that transformers' chat templates take:
    the name and description as they are, and the inputSchema as the function's parameters."""
    return {'type': 'function', 'function': {'name': tool['name'], 'description': tool['description'],
                                             'parameters': tool['inputSchema']}}


# %% [prompt] The prompt
def messages(question, instructions):
    """The server's instructions from its initialize result open the system turn, then how to answer."""
    return [{'role': 'system', 'content': f'{instructions} {ANSWER}'}, {'role': 'user', 'content': question}]


def chat_ids(tokenizer, conversation, schemas):
    """The whole conversation through Qwen3's chat template with the tool schemas, thinking off; the template writes
    each schema as one line of JSON inside <tools></tools> and tells the model to answer calls in <tool_call> tags."""
    return tokenizer.apply_chat_template(conversation, tools=schemas, add_generation_prompt=True, tokenize=True,
                                         enable_thinking=False, return_tensors='pt')[0]


# %% [generate] One reply
@torch.no_grad()
def reply(lm, ids, eos_ids, cfg=Config()):
    """Greedy tokens until an end-of-message token or max_new_tokens, with a key-value cache after the first step.
    Returns (tokens, per-step (top ids, top probabilities), why it stopped)."""
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


# %% [parse] Reading the calls
CALL = re.compile(r'<tool_call>\\s*(.*?)\\s*</tool_call>', re.S)


def parse_calls(text):
    """(calls, answer): every <tool_call> as {name, arguments} (or {error} when its JSON does not parse); with no
    call, the text after the last "Answer:" (or the whole reply)."""
    calls = []
    for body in CALL.findall(text):
        try:
            call = json.loads(body)
            calls.append({'name': call['name'], 'arguments': call.get('arguments', {})})
        except (ValueError, KeyError, TypeError) as error:
            calls.append({'error': f'could not read the call: {error}', 'raw': body})
    found = re.findall(r'Answer:\\s*(.+)', text)
    return calls, None if calls else (found[-1] if found else text).strip()


# %% [client] The client: a session over stdio
def connect(command, args, env=None):
    """The SDK's client for a server it starts as a subprocess and talks to over the subprocess's stdin and stdout.
    mode='legacy' makes it open with the initialize handshake (protocol revisions up to 2025-11-25) rather than the
    SDK's default of first trying the stateless 2026-07-28 revision."""
    return Client(StdioServerParameters(command=command, args=args, env=env), mode='legacy')


def result_text(result):
    """What the model is shown of a tools/call result: its text content (for an error, the server's message)."""
    return '\\n'.join(c.text for c in result.content if c.type == 'text')


# %% [loop] The loop: calls through the server until an answer
async def answer(lm, tokenizer, client, question, schemas, instructions, cfg=Config()):
    """Reply; send each <tool_call> to the server as tools/call and add its result as a tool message, until a reply
    has no call or max_turns. Returns the turns and the answer (None if the loop never ended)."""
    eos = set(lm.generation_config.eos_token_id)
    conversation, turns = messages(question, instructions), []
    for _ in range(cfg.max_turns):
        ids = chat_ids(tokenizer, conversation, schemas)
        tokens, tops, reason = reply(lm, ids, eos, cfg)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        calls, final = parse_calls(text)
        turns.append({'prompt': ids, 'tokens': tokens, 'tops': tops, 'reason': reason, 'text': text, 'calls': calls})
        if not calls:
            conversation.append({'role': 'assistant', 'content': text})
            return turns, final
        good = [c for c in calls if 'error' not in c]
        conversation.append({'role': 'assistant', 'content': CALL.sub('', text).strip(), 'tool_calls': [
            {'type': 'function', 'function': {'name': c['name'], 'arguments': c['arguments']}} for c in good]})
        results = [f"Error: {c['error']}" if 'error' in c else
                   result_text(await client.call_tool(c['name'], c['arguments'])) for c in calls]
        turns[-1]['results'] = results
        conversation += [{'role': 'tool', 'content': r} for r in results]
    return turns, None


# async with connect(python, ["scripts/agents/mcp_server.py"]) as client:
#     tools = [t.model_dump(by_alias=True, exclude_none=True) for t in (await client.list_tools()).tools]
#     turns, final = await answer(lm, tokenizer, client, "On what day of the week was Super Bowl 50 played?",
#                                 [to_schema(t) for t in tools], client.instructions)
`;export{e as default};
