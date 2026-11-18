# FlintFieldfare Agent & Tool Calling Integration

## Native Flint Chat & Tool Protocol

Flint uses a distinct prompt and tool-calling protocol defined in `chat_template.jinja`.

### 1. Role Serialization

| Role | Opening Token | Closing Behavior |
|---|---|---|
| `system` | `<|system|>` | Directly followed by system message text |
| `developer`| `<|developer|>` | Directly followed by developer instruction |
| `user` | `<|user|>` | Followed by user query |
| `assistant`| `<|assistant|>` | Followed by reasoning `<think>...</think>` and text / tool calls |
| `tool` | `<|observation|>`| Wrapped in `<tool_response>...</tool_response>` |

### 2. Thinking / Reasoning

Flint generates reasoning traces within `<think>` and `</think>` tags:
```
<|assistant|>
<think>
I need to examine the directory contents to locate the configuration file.
</think>
I will list the files in the directory.
```

### 3. Native Function Calling Format

Function definitions are provided in the system prompt inside `<tools>...</tools>` as JSON objects.

When invoking a function, the assistant outputs:
```xml
<tool_call>execute_command
<arg_key>command</arg_key><arg_value>ls -la</arg_value>
</tool_call>
```

When multiple tool calls are made, they follow consecutively:
```xml
<tool_call>read_file
<arg_key>path</arg_key><arg_value>config.json</arg_value>
</tool_call>
<tool_call>read_file
<arg_key>path</arg_key><arg_value>Package.swift</arg_value>
</tool_call>
```

The client responds with observation:
```xml
<|observation|><tool_response>{"status": "ok", "content": "..."}</tool_response>
```

### 4. OpenAI Compatibility Translation

The FlintFieldfare HTTP server translates standard OpenAI `tools` and `tool_calls` payloads:
- Inbound: OpenAI `messages` and `tools` array $\to$ Jinja template rendering $\to$ input token IDs.
- Outbound (JSON): Assistant response parsed for `<tool_call>` tags $\to$ OpenAI `tool_calls` array with generated call IDs.
- Outbound (SSE): Streaming tokens emitted with `delta.content` or `delta.tool_calls` structure.
- Never execute tools server-side; execution remains strictly client-side.
