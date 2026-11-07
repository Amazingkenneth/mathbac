"""The Claude backend builds the requests we expect, without calling the API."""

from types import SimpleNamespace as NS

from mathbac.gateway import ToolResult, ToolSpec, redact
from mathbac.gateway.anthropic_backend import FALLBACK_BETA, AnthropicBackend


def block(**kw):
    return NS(**kw)


def response(content, stop="end_turn"):
    return NS(content=content, stop_reason=stop, model="claude-opus-5-5", stop_details=None, _request_id="req_1",
              usage=NS(input_tokens=10, output_tokens=5, cache_read_input_tokens=3, cache_creation_input_tokens=0))


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        return self.responses.pop(0)


def fake_client(responses):
    beta = FakeMessages(responses)
    plain = FakeMessages(responses)
    return NS(beta=NS(messages=beta), messages=plain), beta, plain


TOOLS = [ToolSpec("b_tool", "B", {"type": "object", "properties": {}}), ToolSpec("a_tool", "A", {"type": "object", "properties": {}})]


def test_request_shape_tools_and_fallbacks():
    client, beta, plain = fake_client([
        response([block(type="thinking", thinking=""), block(type="tool_use", id="tu1", name="a_tool", input={"x": 1})], "tool_use"),
        response([block(type="text", text="all done")]),
    ])
    conv = AnthropicBackend(client=client).start(system="SYS", tools=TOOLS, model="claude-opus-5-5", effort="high")
    t1 = conv.send_user("hi")
    assert t1.stop_reason == "tool_use" and t1.tool_calls[0].name == "a_tool"
    req = beta.calls[0]
    assert req["betas"] == [FALLBACK_BETA] and req["fallbacks"] == "default"
    assert req["thinking"] == {"type": "adaptive"}
    assert req["output_config"] == {"effort": "high"}
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert [t["name"] for t in req["tools"]] == ["a_tool", "b_tool"]  # stable order keeps the cache prefix
    assert "temperature" not in req

    t2 = conv.send_tool_results([ToolResult("tu1", "boom", is_error=True)])
    assert t2.text == "all done"
    msgs = beta.calls[1]["messages"]
    # full assistant content (including the thinking block) goes back unchanged
    assert msgs[1]["role"] == "assistant" and msgs[1]["content"][0].type == "thinking"
    assert msgs[2]["content"] == [{"type": "tool_result", "tool_use_id": "tu1", "content": "boom", "is_error": True}]
    assert t2.usage.cache_read_tokens == 3


def test_pause_turn_is_resumed_and_non_fallback_model_uses_plain_endpoint():
    client, beta, plain = fake_client([
        response([block(type="text", text="part 1")], "pause_turn"),
        response([block(type="text", text="part 2")]),
    ])
    conv = AnthropicBackend(client=client).start(system="S", tools=[], model="claude-haiku-5-5")
    t = conv.send_user("go")
    assert t.text == "part 2" and len(plain.calls) == 2 and not beta.calls
    assert t.usage.input_tokens == 20  # usage summed across the paused and resumed calls


def test_redact():
    assert redact("key sk-ant-api03-abcdefghijk and sk-or-v1-zzzzzzzzzzzz") == "key [REDACTED] and [REDACTED]"
    assert redact("my token is hunter2hunter2", ["hunter2hunter2"]) == "my token is [REDACTED]"
