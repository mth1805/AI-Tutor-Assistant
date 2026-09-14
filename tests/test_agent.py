"""Test agent.py: chuẩn hóa content, ask_agent an toàn, stream_agent lọc đúng node."""
from __future__ import annotations

import agent as agent_module


def test_extract_text_from_plain_string():
    assert agent_module._extract_text("xin chào") == "xin chào"


def test_extract_text_from_list_of_dicts():
    content = [{"type": "text", "text": "phần 1"}, {"type": "text", "text": " phần 2"}]
    assert agent_module._extract_text(content) == "phần 1 phần 2"


def test_extract_text_from_none_returns_empty_string():
    assert agent_module._extract_text(None) == ""


class _FakeMessage:
    def __init__(self, content, type_=None, tool_calls=None, name=None):
        self.content = content
        self.type = type_
        self.tool_calls = tool_calls
        self.name = name


class _FakeAgent:
    def invoke(self, payload):
        return {"messages": [_FakeMessage("câu trả lời cuối cùng")]}


def test_ask_agent_returns_final_message_text():
    assert agent_module.ask_agent(_FakeAgent(), "câu hỏi bất kỳ") == "câu trả lời cuối cùng"


class _RaisingAgent:
    def invoke(self, payload):
        raise RuntimeError("lỗi giả lập")


def test_ask_agent_never_raises_and_returns_friendly_message():
    result = agent_module.ask_agent(_RaisingAgent(), "câu hỏi")
    assert isinstance(result, str)
    assert "lỗi" in result.lower()


class _StreamingAgent:
    def stream(self, payload, stream_mode=None):
        yield _FakeMessage("Xin "), {"langgraph_node": "agent"}
        yield _FakeMessage("chào"), {"langgraph_node": "agent"}
        # Bước tool-calling trung gian -> KHÔNG được yield ra ngoài.
        yield _FakeMessage("(dữ liệu tool, không phải câu trả lời)"), {"langgraph_node": "tools"}


def test_stream_agent_yields_only_final_agent_node_chunks():
    chunks = list(agent_module.stream_agent(_StreamingAgent(), "câu hỏi"))
    assert chunks == ["Xin ", "chào"]


class _NoTokenStreamingAgent:
    """Agent giả lập trường hợp không sinh token nào ở node 'agent' (vd toàn
    bộ response chỉ là tool-call) -> phải fallback về ask_agent."""

    def stream(self, payload, stream_mode=None):
        yield _FakeMessage("chỉ có tool-call"), {"langgraph_node": "tools"}

    def invoke(self, payload):
        return {"messages": [_FakeMessage("câu trả lời từ fallback ask_agent")]}


def test_stream_agent_falls_back_to_ask_agent_when_no_tokens_streamed():
    chunks = list(agent_module.stream_agent(_NoTokenStreamingAgent(), "câu hỏi"))
    assert chunks == ["câu trả lời từ fallback ask_agent"]


class _BrokenStreamingAgent:
    def stream(self, payload, stream_mode=None):
        raise RuntimeError("streaming không được hỗ trợ ở phiên bản này")
        yield  # pragma: no cover - không bao giờ tới đây

    def invoke(self, payload):
        return {"messages": [_FakeMessage("câu trả lời không streaming")]}


def test_stream_agent_falls_back_to_ask_agent_on_streaming_error():
    chunks = list(agent_module.stream_agent(_BrokenStreamingAgent(), "câu hỏi"))
    assert chunks == ["câu trả lời không streaming"]
