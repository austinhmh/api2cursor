"""工具定义规范化与参数修复

集中处理旧版散落在各 adapter 中的工具兼容逻辑：
  - 工具定义统一解析为 IRTool（兼容 OpenAI 嵌套、Responses 扁平、Anthropic
    input_schema、Cursor 扁平、OpenAI/Cursor type=custom grammar 五种写法）
  - tool_choice 各协议写法互转
  - LLM 生成参数的常见问题修复：智能引号、file_path→path、StrReplace 精确匹配
  - ApplyPatch / freeform custom 工具：空 schema 合成 + 参数规范化
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from ..core.ir import IRTool, ToolCallBlock

# 智能引号字符集
_SMART_DOUBLE = frozenset('«»\u201c\u201d\u275e\u201f\u201e\u275d')
_SMART_SINGLE = frozenset('\u2018\u2019\u201a\u201b')

_EMPTY_SCHEMA = {'type': 'object', 'properties': {}}

# Cursor/Codex freeform apply_patch 在不支持 grammar 的上游上使用的兼容 schema。
# 与 cliproxy / Codex CLI function 兼容形态一致：单字段 input 字符串。
_FREEFORM_INPUT_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'input': {
            'type': 'string',
            'description': (
                'Full freeform tool input. For ApplyPatch, put the ENTIRE patch text '
                'starting with "*** Begin Patch" and ending with "*** End Patch". '
                'Do not send an empty object.'
            ),
        },
    },
    'required': ['input'],
    'additionalProperties': False,
}

_FREEFORM_DESCRIPTION_HINT = (
    '\n\nIMPORTANT: This is a freeform tool. Put the complete patch text in the '
    '`input` string field. The value must begin with "*** Begin Patch" and end '
    'with "*** End Patch". Never call this tool with an empty input object.'
)

_APPLY_PATCH_NAME_RE = re.compile(r'^(apply_?patch|applypatch)$', re.IGNORECASE)


# ═══════════════════════════════════════════════════════════
#  工具定义解析
# ═══════════════════════════════════════════════════════════


def parse_tool_definitions(tools: Any) -> list[IRTool]:
    """将任意风格的工具定义列表解析为 IRTool 列表。"""
    if not isinstance(tools, list):
        return []
    result: list[IRTool] = []
    for tool in tools:
        parsed = _parse_tool_definition(tool)
        if parsed is not None:
            result.append(parsed)
    return result


def _parse_tool_definition(tool: Any) -> IRTool | None:
    if not isinstance(tool, dict):
        return None

    tool_type = str(tool.get('type') or '').strip().lower()

    # OpenAI 内置 apply_patch：无 JSON schema，按 freeform custom 处理
    if tool_type == 'apply_patch':
        name = str(tool.get('name') or 'apply_patch')
        description = str(tool.get('description') or _default_apply_patch_description())
        return IRTool(
            name=name,
            description=_ensure_freeform_hint(description),
            parameters=dict(_FREEFORM_INPUT_SCHEMA),
            origin='apply_patch',
            custom_format=None,
        )

    # OpenAI / Cursor grammar-based custom tool（Cursor 3.9+ ApplyPatch）
    # 支持扁平 {type:custom,name,format} 与 Chat Completions 嵌套
    # {type:custom, custom:{name,description,format}}
    if tool_type == 'custom':
        nested = tool.get('custom') if isinstance(tool.get('custom'), dict) else {}
        name = str(
            tool.get('name')
            or nested.get('name')
            or tool.get('namespace')
            or nested.get('namespace')
            or ''
        ).strip()
        if not name:
            return None
        description = str(tool.get('description') or nested.get('description') or '')
        custom_format = None
        for source in (tool, nested):
            if isinstance(source.get('format'), dict):
                custom_format = source['format']
                break
        parameters = (
            tool.get('parameters')
            or tool.get('input_schema')
            or nested.get('parameters')
            or nested.get('input_schema')
        )
        if not _schema_has_properties(parameters):
            parameters = dict(_FREEFORM_INPUT_SCHEMA)
            description = _ensure_freeform_hint(description)
        return IRTool(
            name=name,
            description=description,
            parameters=parameters if isinstance(parameters, dict) else dict(_FREEFORM_INPUT_SCHEMA),
            origin='custom',
            custom_format=custom_format,
        )

    # 标准 OpenAI 嵌套格式: {type: "function", function: {name, parameters}}
    if tool.get('type') == 'function' and isinstance(tool.get('function'), dict):
        func = tool['function']
        name = str(func.get('name') or '')
        description = str(func.get('description') or '')
        parameters = func.get('parameters') or _EMPTY_SCHEMA
        origin = 'function'
        # 名字是 ApplyPatch 但 schema 为空：按 freeform 合成
        if _is_apply_patch_name(name) and not _schema_has_properties(parameters):
            parameters = dict(_FREEFORM_INPUT_SCHEMA)
            description = _ensure_freeform_hint(description)
            origin = 'custom'
        return IRTool(
            name=name,
            description=description,
            parameters=parameters if isinstance(parameters, dict) else dict(_EMPTY_SCHEMA),
            origin=origin,
        )

    # Responses 扁平格式: {type: "function", name, parameters} 或
    # Cursor / Anthropic 扁平格式: {name, input_schema | parameters}
    if 'name' in tool:
        name = str(tool.get('name') or '')
        description = str(tool.get('description') or '')
        parameters = tool.get('input_schema') or tool.get('parameters') or _EMPTY_SCHEMA
        origin = 'function'
        if _is_apply_patch_name(name) and not _schema_has_properties(parameters):
            parameters = dict(_FREEFORM_INPUT_SCHEMA)
            description = _ensure_freeform_hint(description)
            origin = 'custom'
        return IRTool(
            name=name,
            description=description,
            parameters=parameters if isinstance(parameters, dict) else dict(_EMPTY_SCHEMA),
            origin=origin,
        )

    return None


def _default_apply_patch_description() -> str:
    return (
        'Use this tool to edit files with a stripped-down patch language. '
        'Wrap every edit in *** Begin Patch / *** End Patch. '
        'Use *** Add File: / *** Update File: / *** Delete File: headers.'
    )


def _ensure_freeform_hint(description: str) -> str:
    text = description or ''
    if 'Put the complete patch text in the' in text or '`input` string field' in text:
        return text
    return text + _FREEFORM_DESCRIPTION_HINT


def _schema_has_properties(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    props = schema.get('properties')
    return isinstance(props, dict) and len(props) > 0


def _is_apply_patch_name(name: str) -> bool:
    return bool(_APPLY_PATCH_NAME_RE.match((name or '').strip()))


def is_custom_tool_name(name: str, tools: list[IRTool] | None = None) -> bool:
    """判断工具名是否应按 custom/freeform 语义处理。"""
    if _is_apply_patch_name(name):
        return True
    if not tools:
        return False
    for tool in tools:
        if tool.name == name and tool.origin in ('custom', 'apply_patch'):
            return True
    return False


def custom_tool_names(tools: list[IRTool] | None) -> set[str]:
    names: set[str] = set()
    for tool in tools or []:
        if tool.origin in ('custom', 'apply_patch') and tool.name:
            names.add(tool.name)
        elif _is_apply_patch_name(tool.name):
            names.add(tool.name)
    return names


def freeform_input_schema() -> dict[str, Any]:
    """返回 freeform custom 工具的兼容 JSON Schema 副本。"""
    return json.loads(json.dumps(_FREEFORM_INPUT_SCHEMA))


def normalize_tools_payload(tools: Any, *, target: str = 'function') -> list[dict[str, Any]]:
    """将原始 tools 列表规范化为上游协议可用的 function / Anthropic 工具定义。

    target:
      - 'function': OpenAI chat / responses function tools
      - 'anthropic': Anthropic messages tools (name + input_schema)
      - 'responses_native': 保留 type=custom（仅当上游真正支持 grammar 时使用）
    """
    parsed = parse_tool_definitions(tools)
    out: list[dict[str, Any]] = []
    for tool in parsed:
        if target == 'responses_native' and tool.origin == 'custom' and tool.custom_format:
            item: dict[str, Any] = {
                'type': 'custom',
                'name': tool.name,
                'description': tool.description,
                'format': tool.custom_format,
            }
            out.append(item)
            continue
        if target == 'anthropic':
            out.append({
                'name': tool.name,
                'description': tool.description,
                'input_schema': tool.parameters or freeform_input_schema(),
            })
        else:
            out.append({
                'type': 'function',
                'name': tool.name,
                'description': tool.description,
                'parameters': tool.parameters or freeform_input_schema(),
            })
    return out


def normalize_tools_in_place(payload: dict[str, Any], *, style: str) -> set[str]:
    """就地规范化 payload['tools']，返回 custom 工具名集合。

    style: 'chat' | 'responses' | 'messages'
    """
    tools = payload.get('tools')
    if not isinstance(tools, list) or not tools:
        return set()

    parsed = parse_tool_definitions(tools)
    custom_names = custom_tool_names(parsed)
    if not parsed:
        return set()

    if style == 'messages':
        payload['tools'] = [
            {
                'name': t.name,
                'description': t.description,
                'input_schema': t.parameters or freeform_input_schema(),
            }
            for t in parsed
        ]
    elif style == 'chat':
        payload['tools'] = [
            {
                'type': 'function',
                'function': {
                    'name': t.name,
                    'description': t.description,
                    'parameters': t.parameters or freeform_input_schema(),
                },
            }
            for t in parsed
        ]
    else:  # responses：上游 cliproxy/Claude 不支持 custom grammar，统一降为 function
        payload['tools'] = [
            {
                'type': 'function',
                'name': t.name,
                'description': t.description,
                'parameters': t.parameters or freeform_input_schema(),
            }
            for t in parsed
        ]
    return custom_names


def parse_tool_choice(tool_choice: Any) -> Any:
    """将各协议的 tool_choice 写法统一为 IR 表示。

    IR 表示: None / 'auto' / 'required' / 'none' / {'name': 工具名}
    """
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice if tool_choice in ('auto', 'required', 'none') else None
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get('type', '')
        if choice_type == 'auto':
            return 'auto'
        if choice_type in ('any', 'required'):
            return 'required'
        if choice_type == 'none':
            return 'none'
        if choice_type in ('function', 'custom'):
            name = tool_choice.get('name') or (tool_choice.get('function') or {}).get('name')
            if name:
                return {'name': name}
        if choice_type == 'tool' and tool_choice.get('name'):
            return {'name': tool_choice['name']}
    return None


# ═══════════════════════════════════════════════════════════
#  参数修复
# ═══════════════════════════════════════════════════════════


def fix_tool_call_block(block: ToolCallBlock, custom_names: set[str] | None = None) -> None:
    """就地修复 IR 工具调用块的参数（非流式响应路径）。"""
    custom_names = custom_names or set()
    is_custom = (
        getattr(block, 'call_style', 'function') == 'custom'
        or block.name in custom_names
        or _is_apply_patch_name(block.name)
    )

    if is_custom:
        block.call_style = 'custom'
        freeform = extract_freeform_input(block.arguments)
        block.arguments = dump_arguments({'input': freeform})
        return

    try:
        args = json.loads(block.arguments) if isinstance(block.arguments, str) else block.arguments
    except (json.JSONDecodeError, ValueError):
        return
    if not isinstance(args, dict):
        return
    args = normalize_args(args)
    args = repair_str_replace_args(block.name, args)
    # ApplyPatch 名字但被当成 function：仍规范化 input
    if _is_apply_patch_name(block.name):
        freeform = extract_freeform_input(args)
        block.call_style = 'custom'
        block.arguments = dump_arguments({'input': freeform})
        return
    block.arguments = json.dumps(args, ensure_ascii=False)


def extract_freeform_input(arguments: Any) -> str:
    """从各种参数形态中提取 freeform 文本。

    支持：
      - 纯字符串 patch 正文
      - {"input": "..."}
      - {"patch": "..."} / {"patchText": "..."} / {"patch_text": "..."}
      - 含真实换行的伪 JSON（流式拼接常见，标准 json.loads 会失败）
      - 其它 JSON 对象：优先取第一个字符串字段，否则 dump 整对象
    """
    if arguments is None:
        return ''
    if isinstance(arguments, dict):
        for key in ('input', 'patch', 'patchText', 'patch_text', 'content', 'text'):
            value = arguments.get(key)
            if isinstance(value, str) and value:
                # 可能再次包了一层 JSON 字符串
                return extract_freeform_input(value) if value.lstrip().startswith('{') else value
            if value is not None and not isinstance(value, (dict, list)):
                return str(value)
        # 单字段对象
        if len(arguments) == 1:
            only = next(iter(arguments.values()))
            if isinstance(only, str):
                return extract_freeform_input(only) if only.lstrip().startswith('{') else only
        try:
            return json.dumps(arguments, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(arguments)

    if not isinstance(arguments, str):
        return str(arguments)

    text = arguments
    stripped = text.strip()
    if not stripped:
        return ''

    # 优先 JSON 解包。注意：含 patch 标记的 {"input":"..."} 也会命中 _looks_like_patch，
    # 必须先解包，否则会把外壳原样返回给 Cursor 导致 Missing header。
    if stripped[0] in '{[':
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            recovered = _extract_freeform_from_broken_json(stripped)
            return recovered if recovered is not None else text
        return extract_freeform_input(parsed)

    # 已经是 patch 正文 / 其它 freeform
    if stripped.startswith('*** Begin Patch') or _looks_like_patch(stripped):
        return text
    return text


def _extract_freeform_from_broken_json(text: str) -> str | None:
    """从含未转义换行的伪 JSON 中提取 input/patch 字段。"""
    for key in ('input', 'patch', 'patchText', 'patch_text', 'content', 'text'):
        key_pat = f'"{key}"'
        idx = text.find(key_pat)
        if idx < 0:
            continue
        colon = text.find(':', idx + len(key_pat))
        if colon < 0:
            continue
        i = colon + 1
        while i < len(text) and text[i] in ' \t\r\n':
            i += 1
        if i >= len(text) or text[i] != '"':
            continue
        i += 1  # skip opening quote
        out: list[str] = []
        while i < len(text):
            ch = text[i]
            if ch == '\\' and i + 1 < len(text):
                out.append(ch)
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                raw = ''.join(out)
                # raw 可能是合法 JSON 字符串内容（含 \\n 转义），也可能含真实换行
                try:
                    return json.loads('"' + raw + '"')
                except (json.JSONDecodeError, ValueError):
                    pass
                result: list[str] = []
                j = 0
                while j < len(raw):
                    if raw[j] == '\\' and j + 1 < len(raw):
                        nxt = raw[j + 1]
                        if nxt == 'n':
                            result.append('\n')
                        elif nxt == 't':
                            result.append('\t')
                        elif nxt == 'r':
                            result.append('\r')
                        elif nxt == '"':
                            result.append('"')
                        elif nxt == '\\':
                            result.append('\\')
                        else:
                            result.append(nxt)
                        j += 2
                    else:
                        result.append(raw[j])
                        j += 1
                return ''.join(result)
            out.append(ch)
            i += 1
    return None

def unwrap_custom_tool_input(arguments: str) -> str:
    """从 {"input":"..."} 参数中取出 freeform 正文，供 custom_tool_call 回写。"""
    return extract_freeform_input(arguments)


def normalize_args(args: dict[str, Any]) -> dict[str, Any]:
    """规范化工具参数：file_path → path。"""
    if isinstance(args, dict) and 'file_path' in args and 'path' not in args:
        args['path'] = args.pop('file_path')
    return args


def repair_str_replace_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """修复 StrReplace/search_replace 工具的精确匹配问题。

    当 old_string 包含智能引号导致无法精确匹配文件内容时，
    用容错正则在文件中查找唯一匹配并替换为实际内容。
    """
    if not isinstance(args, dict):
        return args

    name_lower = (tool_name or '').lower()
    if 'str_replace' not in name_lower and 'search_replace' not in name_lower:
        return args

    old_str = args.get('old_string') or args.get('old_str')
    if not old_str:
        return args

    file_path = args.get('path') or args.get('file_path')
    if not file_path or not os.path.isfile(file_path):
        return args

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except OSError:
        return args

    # 已精确匹配，无需修复
    if old_str in content:
        return args

    pattern = _build_fuzzy_pattern(old_str)
    try:
        matches = list(re.finditer(pattern, content))
    except re.error:
        return args

    # 仅在唯一匹配时修复，避免歧义
    if len(matches) != 1:
        return args

    matched = matches[0].group()
    if 'old_string' in args:
        args['old_string'] = matched
    elif 'old_str' in args:
        args['old_str'] = matched

    # 同步修复 new_string 中的智能引号
    new_str = args.get('new_string') or args.get('new_str')
    if new_str:
        fixed = replace_smart_quotes(new_str)
        if 'new_string' in args:
            args['new_string'] = fixed
        elif 'new_str' in args:
            args['new_str'] = fixed

    return args


def replace_smart_quotes(text: str) -> str:
    """将智能引号替换为普通 ASCII 引号。"""
    return ''.join(
        '"' if ch in _SMART_DOUBLE else
        "'" if ch in _SMART_SINGLE else
        ch for ch in text
    )


def parse_arguments_dict(arguments: Any) -> dict[str, Any]:
    """将工具参数尽量解析为对象，供需要结构化 input 的协议使用。"""
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return {}
    stripped = arguments.strip()
    if not stripped:
        return {}
    # freeform patch 正文不是 JSON：包装成 {"input": "..."}
    if stripped[0] not in '{[':
        if '*** Begin Patch' in arguments or _looks_like_patch(arguments):
            return {'input': arguments}
        return {}
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, ValueError):
        if '*** Begin Patch' in arguments or _looks_like_patch(arguments):
            return {'input': arguments}
        return {}
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, str):
        return {'input': parsed}
    return {}


def _looks_like_patch(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            '*** Add File:',
            '*** Update File:',
            '*** Delete File:',
            '*** End Patch',
        )
    )


def dump_arguments(arguments: Any) -> str:
    """将工具参数统一序列化为 JSON 字符串。

    注意：普通 function 工具的 arguments 若已是 JSON 字符串则原样返回；
    只有看起来像 ApplyPatch freeform 正文时，才包装为 {"input": "..."}。
    """
    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return '{}'
        if stripped.startswith('{') or stripped.startswith('['):
            return arguments
        if '*** Begin Patch' in arguments or _looks_like_patch(arguments):
            try:
                return json.dumps({'input': arguments}, ensure_ascii=False)
            except (TypeError, ValueError):
                return '{}'
        # 其它自由文本：原样返回，避免破坏非 custom 工具
        return arguments
    try:
        return json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)
    except (TypeError, ValueError):
        return '{}'


def _build_fuzzy_pattern(text: str) -> str:
    """构建容错正则：智能引号可互换、空白可伸缩、反斜杠可重复。"""
    parts = []
    for ch in text:
        if ch in _SMART_DOUBLE or ch == '"':
            parts.append('["\u00ab\u201c\u201d\u275e\u201f\u201e\u275d\u00bb]')
        elif ch in _SMART_SINGLE or ch == "'":
            parts.append("['\u2018\u2019\u201a\u201b]")
        elif ch in (' ', '\t'):
            parts.append(r'\s+')
        elif ch == '\\':
            parts.append(r'\\{1,2}')
        else:
            parts.append(re.escape(ch))
    return ''.join(parts)


# ═══════════════════════════════════════════════════════════
#  兼容别名（供 protocols 层调用）
# ═══════════════════════════════════════════════════════════


def is_custom_origin_tool(tool: IRTool | None = None, *, name: str = '', origin: str = '') -> bool:
    """判断工具是否应按 custom/ApplyPatch freeform 语义处理。"""
    if tool is not None:
        if getattr(tool, 'origin', 'function') in ('custom', 'apply_patch'):
            return True
        name = tool.name or name
        origin = getattr(tool, 'origin', '') or origin
    if origin in ('custom', 'apply_patch'):
        return True
    return _is_apply_patch_name(name)


def extract_custom_input(arguments: Any) -> str:
    """extract_freeform_input 的别名，供协议层回写 custom_tool_call.input。"""
    return extract_freeform_input(arguments)


def repair_custom_tool_args(
    tool_name: str,
    args: dict[str, Any],
    *,
    call_style: str = 'function',
) -> dict[str, Any]:
    """把 custom/ApplyPatch 参数统一到 {"input": "..."}。"""
    if not isinstance(args, dict):
        return {'input': extract_freeform_input(args)}
    if not (is_custom_origin_tool(name=tool_name) or call_style == 'custom'):
        # 内容像 patch 时仍规范化
        if not any(
            isinstance(v, str) and ('*** Begin Patch' in v or _looks_like_patch(v))
            for v in args.values()
        ):
            return args
    freeform = extract_freeform_input(args)
    return {'input': freeform}


# ═══════════════════════════════════════════════════════════
#  Responses raw 透传：function_call → custom_tool_call 回写
# ═══════════════════════════════════════════════════════════


def collect_custom_tool_names_from_payload(payload: dict[str, Any] | None) -> set[str]:
    """从原始请求 payload 的 tools 字段提取 custom/ApplyPatch 工具名。"""
    if not isinstance(payload, dict):
        return set()
    return custom_tool_names(parse_tool_definitions(payload.get('tools')))


def rewrite_responses_item_for_custom_tools(
    item: Any,
    custom_names: set[str],
) -> Any:
    """把 output/input 中的 function_call 项还原为 custom_tool_call。"""
    if not isinstance(item, dict) or not custom_names:
        return item
    item_type = item.get('type', '')
    name = item.get('name') or ''
    if item_type == 'function_call' and (name in custom_names or is_custom_origin_tool(name=name)):
        freeform = extract_freeform_input(item.get('arguments', ''))
        return {
            'id': item.get('id') or '',
            'type': 'custom_tool_call',
            'status': item.get('status') or 'completed',
            'call_id': item.get('call_id') or item.get('id') or '',
            'name': name,
            'input': freeform,
        }
    if item_type == 'function_call_output' and item.get('call_id'):
        # 输出项本身无 name；若调用侧已是 custom，保持原样即可
        return item
    return item


def rewrite_responses_payload_for_custom_tools(
    payload: dict[str, Any],
    custom_names: set[str],
) -> dict[str, Any]:
    """非流式 / completed 事件：重写 response.output 中的 function_call。"""
    if not custom_names or not isinstance(payload, dict):
        return payload

    response_obj = payload.get('response')
    if isinstance(response_obj, dict) and isinstance(response_obj.get('output'), list):
        response_obj['output'] = [
            rewrite_responses_item_for_custom_tools(item, custom_names)
            for item in response_obj['output']
        ]

    if isinstance(payload.get('output'), list):
        payload['output'] = [
            rewrite_responses_item_for_custom_tools(item, custom_names)
            for item in payload['output']
        ]

    # output_item.added / done 的 item 字段
    item = payload.get('item')
    if isinstance(item, dict):
        payload['item'] = rewrite_responses_item_for_custom_tools(item, custom_names)

    return payload


class ResponsesCustomToolStreamRewriter:
    """流式透传时把 ApplyPatch 等 custom 工具的 function_call 事件还原为 custom_tool_call。

    上游（cliproxy / chat 中转）通常输出：
      response.output_item.added  item.type=function_call
      response.function_call_arguments.delta/done
      response.output_item.done   item.type=function_call

    Cursor 本地 ApplyPatch 执行器需要：
      response.output_item.added  item.type=custom_tool_call, input=""
      response.custom_tool_call_input.delta/done  input=freeform
      response.output_item.done   item.type=custom_tool_call, input=freeform
    """

    def __init__(self, custom_names: set[str] | None = None):
        self._custom_names = set(custom_names or set())
        # call_id / item_id → 缓冲
        self._buffers: dict[str, dict[str, Any]] = {}

    def _is_custom_name(self, name: str) -> bool:
        return bool(name) and (name in self._custom_names or is_custom_origin_tool(name=name))

    def _key_for(self, payload: dict[str, Any], item: dict[str, Any] | None = None) -> str:
        item = item or {}
        return str(
            payload.get('item_id')
            or item.get('id')
            or item.get('call_id')
            or payload.get('call_id')
            or payload.get('output_index')
            or ''
        )

    def process(self, event_type: str, payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        """返回零或多条 (event_type, payload) 供下游发送。"""
        if not self._custom_names:
            return [(event_type, payload)]

        etype = event_type or payload.get('type', '')
        item = payload.get('item') if isinstance(payload.get('item'), dict) else None

        # 1) output_item.added: function_call → custom_tool_call
        if etype == 'response.output_item.added' and item:
            name = item.get('name') or ''
            if item.get('type') == 'function_call' and self._is_custom_name(name):
                key = self._key_for(payload, item)
                self._buffers[key] = {
                    'name': name,
                    'call_id': item.get('call_id') or '',
                    'item_id': item.get('id') or '',
                    'args': item.get('arguments') or '',
                    'output_index': payload.get('output_index'),
                }
                new_item = {
                    'id': item.get('id') or '',
                    'type': 'custom_tool_call',
                    'status': item.get('status') or 'in_progress',
                    'call_id': item.get('call_id') or '',
                    'name': name,
                    'input': '',
                }
                new_payload = dict(payload)
                new_payload['type'] = 'response.output_item.added'
                new_payload['item'] = new_item
                return [('response.output_item.added', new_payload)]
            return [(etype, payload)]

        # 2) function_call_arguments.delta
        if etype == 'response.function_call_arguments.delta':
            key = self._key_for(payload)
            buf = self._buffers.get(key)
            # 也尝试用 output_index 匹配
            if buf is None:
                for b in self._buffers.values():
                    if b.get('output_index') == payload.get('output_index'):
                        buf = b
                        key = self._key_for({'item_id': b.get('item_id')})
                        break
            if buf is None:
                # 未知是否 custom：原样透传
                return [(etype, payload)]
            delta = payload.get('delta') or ''
            buf['args'] = (buf.get('args') or '') + delta
            # 流式 JSON 增量无法可靠剥壳；缓冲到 done 再一次性输出 freeform
            return []

        # 3) function_call_arguments.done
        if etype == 'response.function_call_arguments.done':
            key = self._key_for(payload)
            buf = self._buffers.get(key)
            if buf is None:
                for k, b in self._buffers.items():
                    if b.get('output_index') == payload.get('output_index'):
                        buf = b
                        key = k
                        break
            if buf is None:
                return [(etype, payload)]
            args = payload.get('arguments')
            if args is None:
                args = buf.get('args') or ''
            freeform = extract_freeform_input(args)
            buf['args'] = args
            buf['freeform'] = freeform
            new_payload = {
                'type': 'response.custom_tool_call_input.done',
                'item_id': payload.get('item_id') or buf.get('item_id') or '',
                'output_index': payload.get('output_index', buf.get('output_index')),
                'input': freeform,
            }
            if 'sequence_number' in payload:
                new_payload['sequence_number'] = payload['sequence_number']
            return [('response.custom_tool_call_input.done', new_payload)]

        # 4) output_item.done
        if etype == 'response.output_item.done' and item:
            name = item.get('name') or ''
            if item.get('type') == 'function_call' and self._is_custom_name(name):
                key = self._key_for(payload, item)
                buf = self._buffers.get(key)
                args = item.get('arguments') or (buf.get('args') if buf else '') or ''
                freeform = (buf.get('freeform') if buf else None) or extract_freeform_input(args)
                new_item = {
                    'id': item.get('id') or '',
                    'type': 'custom_tool_call',
                    'status': item.get('status') or 'completed',
                    'call_id': item.get('call_id') or '',
                    'name': name,
                    'input': freeform,
                }
                new_payload = dict(payload)
                new_payload['type'] = 'response.output_item.done'
                new_payload['item'] = new_item
                return [('response.output_item.done', new_payload)]
            return [(etype, payload)]

        # 5) completed / incomplete：重写 output 数组
        if etype in ('response.completed', 'response.incomplete', 'response.failed'):
            rewrite_responses_payload_for_custom_tools(payload, self._custom_names)
            return [(etype, payload)]

        # 其它事件：若 payload 内嵌 item/output，也尝试重写
        rewrite_responses_payload_for_custom_tools(payload, self._custom_names)
        return [(etype, payload)]
