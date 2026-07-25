"""ApplyPatch / type=custom 兼容层单元测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.compat.tools import (
    extract_custom_input,
    fix_tool_call_block,
    normalize_tools_in_place,
    parse_tool_definitions,
    repair_custom_tool_args,
)
from app.core.ir import IRResponse, IRRequest, ToolCallBlock
from app.protocols.anthropic import AnthropicCodec
from app.protocols.chat_completions import ChatCompletionsCodec
from app.protocols.responses_api import ResponsesCodec

SAMPLE_PATCH = """*** Begin Patch
*** Add File: hello.txt
+hello world
*** End Patch
"""

CUSTOM_TOOL = {
    "type": "custom",
    "name": "ApplyPatch",
    "description": "Use this tool to edit files",
    "format": {
        "type": "grammar",
        "syntax": "lark",
        "definition": 'start: begin_patch hunk end_patch\nbegin_patch: "*** Begin Patch" NEWLINE',
    },
}


def test_parse_custom_applypatch_synthesizes_input_schema():
    tools = parse_tool_definitions([CUSTOM_TOOL])
    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "ApplyPatch"
    assert tool.origin == "custom"
    assert tool.parameters["properties"]["input"]["type"] == "string"
    assert "input" in tool.parameters["required"]
    assert tool.custom_format and tool.custom_format.get("type") == "grammar"


def test_parse_empty_function_applypatch():
    tools = parse_tool_definitions(
        [
            {
                "type": "function",
                "function": {
                    "name": "ApplyPatch",
                    "description": "edit",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    )
    assert tools[0].origin == "custom"
    assert "input" in tools[0].parameters["properties"]


def test_parse_apply_patch_builtin():
    tools = parse_tool_definitions([{"type": "apply_patch"}])
    assert tools[0].origin == "apply_patch"
    assert "input" in tools[0].parameters["properties"]


def test_normalize_tools_messages_style():
    payload = {"tools": [CUSTOM_TOOL]}
    names = normalize_tools_in_place(payload, style="messages")
    assert "ApplyPatch" in names
    tool = payload["tools"][0]
    assert tool["name"] == "ApplyPatch"
    assert tool["input_schema"]["properties"]["input"]["type"] == "string"


def test_normalize_tools_responses_downgrades_to_function():
    payload = {"tools": [CUSTOM_TOOL]}
    normalize_tools_in_place(payload, style="responses")
    tool = payload["tools"][0]
    assert tool["type"] == "function"
    assert tool["parameters"]["properties"]["input"]["type"] == "string"


def test_fix_tool_call_block_wraps_freeform():
    block = ToolCallBlock(name="ApplyPatch", arguments=SAMPLE_PATCH)
    fix_tool_call_block(block)
    assert block.call_style == "custom"
    args = json.loads(block.arguments)
    assert args["input"].startswith("*** Begin Patch")


def test_fix_tool_call_block_alias_fields():
    block = ToolCallBlock(
        name="ApplyPatch",
        arguments=json.dumps({"patch": SAMPLE_PATCH}),
    )
    fix_tool_call_block(block)
    assert json.loads(block.arguments)["input"].startswith("*** Begin Patch")


def test_extract_custom_input():
    assert extract_custom_input({"input": SAMPLE_PATCH}).startswith("*** Begin Patch")
    assert extract_custom_input(SAMPLE_PATCH).startswith("*** Begin Patch")
    assert extract_custom_input(json.dumps({"patch": SAMPLE_PATCH})).startswith(
        "*** Begin Patch"
    )


def test_anthropic_upstream_tools_have_input_schema():
    codec = AnthropicCodec()
    req = IRRequest(
        model="claude",
        tools=parse_tool_definitions([CUSTOM_TOOL]),
        messages=[],
    )
    body = codec.build_request(req, "claude-sonnet-5")
    tool = body["tools"][0]
    assert tool["name"] == "ApplyPatch"
    assert tool["input_schema"]["properties"]["input"]["type"] == "string"


def test_responses_client_build_custom_tool_call():
    codec = ResponsesCodec()
    codec.set_custom_tool_names({"ApplyPatch"})
    response = IRResponse(
        blocks=[
            ToolCallBlock(
                id="call_1",
                name="ApplyPatch",
                arguments=json.dumps({"input": SAMPLE_PATCH}),
                call_style="custom",
            )
        ],
        finish_reason="tool_calls",
    )
    body = codec.build_response(response, "gpt-5.4")
    item = body["output"][0]
    assert item["type"] == "custom_tool_call"
    assert item["input"].startswith("*** Begin Patch")


def test_chat_client_build_custom_tool_call():
    codec = ChatCompletionsCodec()
    response = IRResponse(
        blocks=[
            ToolCallBlock(
                id="call_1",
                name="ApplyPatch",
                arguments=json.dumps({"input": SAMPLE_PATCH}),
                call_style="custom",
            )
        ],
        finish_reason="tool_calls",
    )
    body = codec.build_response(response, "gpt-5.4")
    tc = body["choices"][0]["message"]["tool_calls"][0]
    assert tc["type"] == "custom"
    assert tc["custom"]["input"].startswith("*** Begin Patch")


def test_repair_custom_tool_args():
    fixed = repair_custom_tool_args("ApplyPatch", {"diff": SAMPLE_PATCH})
    assert fixed["input"].startswith("*** Begin Patch")


def test_raw_responses_stream_rewriter_function_to_custom():
    from app.compat.tools import ResponsesCustomToolStreamRewriter

    rewriter = ResponsesCustomToolStreamRewriter({'ApplyPatch'})
    added = {
        'type': 'response.output_item.added',
        'output_index': 2,
        'item': {
            'id': 'fc_call_1',
            'type': 'function_call',
            'status': 'in_progress',
            'call_id': 'call_1',
            'name': 'ApplyPatch',
            'arguments': '',
        },
        'sequence_number': 1,
    }
    out = rewriter.process('response.output_item.added', added)
    assert len(out) == 1
    assert out[0][0] == 'response.output_item.added'
    assert out[0][1]['item']['type'] == 'custom_tool_call'
    assert out[0][1]['item']['input'] == ''

    # delta is buffered (JSON fragments not emitted as freeform)
    delta = {
        'type': 'response.function_call_arguments.delta',
        'item_id': 'fc_call_1',
        'output_index': 2,
        'delta': '{"input":"' + SAMPLE_PATCH.replace('\n', '\\n') + '"}',
        'sequence_number': 2,
    }
    assert rewriter.process('response.function_call_arguments.delta', delta) == []

    done = {
        'type': 'response.function_call_arguments.done',
        'item_id': 'fc_call_1',
        'output_index': 2,
        'arguments': json.dumps({'input': SAMPLE_PATCH}),
        'sequence_number': 3,
    }
    out = rewriter.process('response.function_call_arguments.done', done)
    assert len(out) == 1
    assert out[0][0] == 'response.custom_tool_call_input.done'
    assert out[0][1]['input'].startswith('*** Begin Patch')
    assert '{\"input\"' not in out[0][1]['input']

    item_done = {
        'type': 'response.output_item.done',
        'output_index': 2,
        'item': {
            'id': 'fc_call_1',
            'type': 'function_call',
            'status': 'completed',
            'call_id': 'call_1',
            'name': 'ApplyPatch',
            'arguments': json.dumps({'input': SAMPLE_PATCH}),
        },
        'sequence_number': 4,
    }
    out = rewriter.process('response.output_item.done', item_done)
    assert out[0][1]['item']['type'] == 'custom_tool_call'
    assert out[0][1]['item']['input'].startswith('*** Begin Patch')


def test_rewrite_completed_payload_output_array():
    from app.compat.tools import rewrite_responses_payload_for_custom_tools

    payload = {
        'type': 'response.completed',
        'response': {
            'id': 'resp_1',
            'output': [
                {
                    'id': 'fc_1',
                    'type': 'function_call',
                    'status': 'completed',
                    'call_id': 'call_1',
                    'name': 'ApplyPatch',
                    'arguments': json.dumps({'input': SAMPLE_PATCH}),
                }
            ],
        },
    }
    rewrite_responses_payload_for_custom_tools(payload, {'ApplyPatch'})
    item = payload['response']['output'][0]
    assert item['type'] == 'custom_tool_call'
    assert item['input'].startswith('*** Begin Patch')
    assert 'arguments' not in item


def test_collect_custom_names_from_payload():
    from app.compat.tools import collect_custom_tool_names_from_payload

    names = collect_custom_tool_names_from_payload({
        'tools': [CUSTOM_TOOL, {'type': 'function', 'name': 'Shell', 'parameters': {'type': 'object', 'properties': {}}}]
    })
    assert names == {'ApplyPatch'}


def test_chat_stream_encoder_emits_type_custom_for_applypatch():
    from app.core.ir import ToolCallStart, ToolCallDelta, ToolCallEnd, StreamEnd
    from app.protocols.chat_completions import ChatStreamEncoder
    import json as _json

    enc = ChatStreamEncoder('gpt-5.2', custom_tool_names={'ApplyPatch'})
    out = []
    out += enc.encode(ToolCallStart(index=0, id='call_1', name='ApplyPatch', call_style='custom'))
    # JSON shell deltas must be buffered
    out += enc.encode(ToolCallDelta(index=0, arguments='{"input":' + _json.dumps(SAMPLE_PATCH)[0:10]))
    out += enc.encode(ToolCallDelta(index=0, arguments=_json.dumps(SAMPLE_PATCH)[10:] + '}'))
    out += enc.encode(ToolCallEnd(
        index=0, id='call_1', name='ApplyPatch',
        arguments=_json.dumps({'input': SAMPLE_PATCH}),
        call_style='custom',
    ))
    out += enc.encode(StreamEnd(finish_reason='tool_calls'))

    payloads = []
    for chunk in out:
        if not chunk.startswith('data: '):
            continue
        body = chunk[len('data: '):].strip()
        if body == '[DONE]':
            continue
        payloads.append(_json.loads(body))

    # find tool_calls
    customs = []
    for p in payloads:
        delta = (p.get('choices') or [{}])[0].get('delta') or {}
        for tc in delta.get('tool_calls') or []:
            if tc.get('type') == 'custom' or 'custom' in tc:
                customs.append(tc)
    assert customs, f'no custom tool_calls in {payloads}'
    # last custom should have freeform input
    last = customs[-1]
    assert last['type'] == 'custom'
    assert last['custom']['name'] == 'ApplyPatch'
    assert last['custom']['input'].startswith('*** Begin Patch')
    assert not last['custom']['input'].lstrip().startswith('{')


def test_chat_stream_encoder_detects_applypatch_by_name_without_custom_set():
    from app.core.ir import ToolCallStart, ToolCallEnd
    from app.protocols.chat_completions import ChatStreamEncoder
    import json as _json

    enc = ChatStreamEncoder('gpt-5.2')  # no explicit set
    out = enc.encode(ToolCallStart(index=0, id='c1', name='ApplyPatch'))
    out += enc.encode(ToolCallEnd(
        index=0, id='c1', name='ApplyPatch',
        arguments=_json.dumps({'input': SAMPLE_PATCH}),
    ))
    joined = ''.join(out)
    assert '"type": "custom"' in joined or '"type":"custom"' in joined
    assert '*** Begin Patch' in joined
