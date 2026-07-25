"""ApplyPatch / type=custom 兼容层测试。"""

from __future__ import annotations

import json

from app.compat.tools import (
    custom_tool_names,
    extract_freeform_input,
    fix_tool_call_block,
    freeform_input_schema,
    normalize_tools_in_place,
    parse_tool_definitions,
    unwrap_custom_tool_input,
)
from app.core.ir import ToolCallBlock
from app.protocols.responses_api import ResponsesCodec


def test_parse_type_custom_apply_patch_synthesizes_input_schema():
    tools = parse_tool_definitions([
        {
            'type': 'custom',
            'name': 'ApplyPatch',
            'description': 'edit files',
            'format': {
                'type': 'grammar',
                'syntax': 'lark',
                'definition': 'start: begin_patch',
            },
        },
        {
            'type': 'function',
            'name': 'Shell',
            'description': 'run shell',
            'parameters': {
                'type': 'object',
                'properties': {'command': {'type': 'string'}},
                'required': ['command'],
            },
        },
    ])
    assert len(tools) == 2
    apply_tool = tools[0]
    assert apply_tool.name == 'ApplyPatch'
    assert apply_tool.origin == 'custom'
    assert apply_tool.custom_format is not None
    assert apply_tool.parameters['properties']['input']['type'] == 'string'
    assert 'input' in apply_tool.parameters['required']
    assert tools[1].name == 'Shell'
    assert tools[1].origin == 'function'


def test_parse_empty_function_applypatch_also_synthesizes():
    tools = parse_tool_definitions([
        {
            'name': 'ApplyPatch',
            'description': 'edit',
            'input_schema': {'type': 'object', 'properties': {}},
        }
    ])
    assert tools[0].origin == 'custom'
    assert tools[0].parameters['properties']['input']['type'] == 'string'


def test_normalize_tools_in_place_messages_and_responses():
    payload = {
        'tools': [
            {
                'type': 'custom',
                'name': 'ApplyPatch',
                'description': 'edit',
                'format': {'type': 'grammar', 'syntax': 'lark', 'definition': 'start: x'},
            }
        ]
    }
    names = normalize_tools_in_place(payload, style='messages')
    assert 'ApplyPatch' in names
    tool = payload['tools'][0]
    assert tool['name'] == 'ApplyPatch'
    assert tool['input_schema']['properties']['input']['type'] == 'string'
    assert '*** Begin Patch' in tool['description'] or 'input' in tool['description']

    payload2 = {
        'tools': [
            {
                'type': 'custom',
                'name': 'ApplyPatch',
                'description': 'edit',
                'format': {'type': 'grammar', 'syntax': 'lark', 'definition': 'start: x'},
            }
        ]
    }
    normalize_tools_in_place(payload2, style='responses')
    tool2 = payload2['tools'][0]
    assert tool2['type'] == 'function'
    assert tool2['parameters']['properties']['input']['type'] == 'string'


def test_fix_tool_call_block_wraps_freeform():
    block = ToolCallBlock(
        id='call_1',
        name='ApplyPatch',
        arguments='*** Begin Patch\n*** Add File: a.txt\n+hi\n*** End Patch\n',
        call_style='function',
    )
    fix_tool_call_block(block)
    assert block.call_style == 'custom'
    args = json.loads(block.arguments)
    assert args['input'].startswith('*** Begin Patch')


def test_extract_freeform_input_aliases():
    assert extract_freeform_input({'input': 'PATCH'}) == 'PATCH'
    assert extract_freeform_input({'patchText': 'PATCH'}) == 'PATCH'
    assert extract_freeform_input('{"input":"PATCH"}') == 'PATCH'
    assert unwrap_custom_tool_input(json.dumps({'input': 'X'})) == 'X'


def test_responses_codec_build_response_emits_custom_tool_call():
    codec = ResponsesCodec()
    codec.set_custom_tool_names({'ApplyPatch'})
    from app.core.ir import IRResponse

    response = IRResponse(
        id='resp_1',
        model='gpt',
        blocks=[
            ToolCallBlock(
                id='call_1',
                name='ApplyPatch',
                arguments=json.dumps({'input': '*** Begin Patch\n*** Add File: a.md\n+x\n*** End Patch\n'}),
                call_style='custom',
            )
        ],
        finish_reason='tool_calls',
    )
    out = codec.build_response(response, 'gpt-client')
    assert out['output'][0]['type'] == 'custom_tool_call'
    assert out['output'][0]['input'].startswith('*** Begin Patch')
    assert 'arguments' not in out['output'][0]


def test_responses_codec_build_request_uses_function_schema():
    codec = ResponsesCodec()
    payload = {
        'model': 'gpt',
        'input': 'hi',
        'tools': [
            {
                'type': 'custom',
                'name': 'ApplyPatch',
                'description': 'edit files',
                'format': {'type': 'grammar', 'syntax': 'lark', 'definition': 'start: x'},
            }
        ],
    }
    request = codec.parse_request(payload)
    built = codec.build_request(request, 'upstream-model')
    tool = built['tools'][0]
    assert tool['type'] == 'function'
    assert tool['name'] == 'ApplyPatch'
    assert tool['parameters']['properties']['input']['type'] == 'string'


def test_freeform_schema_shape():
    schema = freeform_input_schema()
    assert schema['required'] == ['input']
    assert 'input' in schema['properties']


def test_custom_tool_names_helper():
    tools = parse_tool_definitions([
        {'type': 'custom', 'name': 'ApplyPatch', 'description': 'x'},
        {'type': 'function', 'name': 'Shell', 'parameters': {'type': 'object', 'properties': {}}},
    ])
    names = custom_tool_names(tools)
    assert names == {'ApplyPatch'}
