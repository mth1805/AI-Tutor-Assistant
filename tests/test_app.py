"""Test app.py: chỉ nhắm vào LOGIC thuần (workspace resolution, bootstrap
fail-fast, cache agent, tải trạng thái workspace) — không test phần vẽ UI
(layout cột, nút toggle ẩn/hiện xem trước) vì đó là phạm vi test tích hợp
(xem README, mục Testing)."""
from __future__ import annotations

import streamlit as st

import app
from tests.conftest import _FakeStopException


def setup_function():
    """Reset session_state/query_params sạch trước mỗi test, tránh test
    trước ảnh hưởng test sau (module streamlit là singleton dùng chung)."""
    st.session_state.clear()
    st.query_params.clear()


# ---------------------------------------------------------------------------
# _get_workspace_id
# ---------------------------------------------------------------------------


def test_get_workspace_id_reuses_existing_query_param():
    st.query_params["ws"] = "ws_abc123"
    assert app._get_workspace_id() == "ws_abc123"


def test_get_workspace_id_creates_new_when_missing():
    assert "ws" not in st.query_params
    ws = app._get_workspace_id()
    assert ws.startswith("ws_")
    # Phải gắn ngược vào URL để refresh trang không sinh workspace mới.
    assert st.query_params["ws"] == ws


def test_get_workspace_id_generates_different_ids_each_time_when_missing():
    ws1 = app._get_workspace_id()
    st.query_params.clear()
    ws2 = app._get_workspace_id()
    assert ws1 != ws2


# ---------------------------------------------------------------------------
# _bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_stops_on_config_error(monkeypatch):
    monkeypatch.setattr(app, "get_config_error", lambda: "Thiếu GOOGLE_API_KEY")
    try:
        app._bootstrap()
        raised = False
    except _FakeStopException:
        raised = True
    # st.stop() phải được gọi ngay khi có lỗi cấu hình — giống hệt hành vi
    # thật (script dừng lại, người dùng chỉ thấy thông báo lỗi, không thấy gì
    # thêm phía sau).
    assert raised


def test_bootstrap_skips_db_check_when_already_checked(monkeypatch):
    monkeypatch.setattr(app, "get_config_error", lambda: None)
    st.session_state["db_checked"] = True

    called = []
    monkeypatch.setattr(app, "check_database_connection", lambda: called.append(1))

    assert app._bootstrap() is True
    assert called == []  # không gọi lại check_database_connection nếu đã check rồi


def test_bootstrap_checks_db_and_caches_result(monkeypatch):
    monkeypatch.setattr(app, "get_config_error", lambda: None)
    monkeypatch.setattr(app, "check_database_connection", lambda: None)

    assert app._bootstrap() is True
    assert st.session_state["db_checked"] is True


def test_bootstrap_stops_when_db_check_raises(monkeypatch):
    from exceptions import DatabaseConnectionError

    monkeypatch.setattr(app, "get_config_error", lambda: None)

    def _boom():
        raise DatabaseConnectionError("không kết nối được Postgres")

    monkeypatch.setattr(app, "check_database_connection", _boom)

    try:
        app._bootstrap()
        raised = False
    except _FakeStopException:
        raised = True
    assert raised
    assert "db_checked" not in st.session_state


# ---------------------------------------------------------------------------
# _load_workspace_state
# ---------------------------------------------------------------------------


def test_load_workspace_state_loads_history_and_index_status(monkeypatch):
    monkeypatch.setattr(app, "get_chat_history", lambda ws: [{"role": "user", "content": "xin chào"}])
    monkeypatch.setattr(app, "collection_has_documents", lambda ws: True)

    app._load_workspace_state("ws_abc")

    assert st.session_state["messages"] == [{"role": "user", "content": "xin chào"}]
    assert st.session_state["is_indexed"] is True
    assert st.session_state["loaded_workspace"] == "ws_abc"


def test_load_workspace_state_skips_reload_for_same_workspace(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "get_chat_history", lambda ws: calls.append(ws) or [])
    monkeypatch.setattr(app, "collection_has_documents", lambda ws: True)

    app._load_workspace_state("ws_abc")
    calls.clear()
    app._load_workspace_state("ws_abc")  # cùng workspace -> không tải lại

    assert calls == []


def test_load_workspace_state_reloads_when_workspace_changes(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "get_chat_history", lambda ws: calls.append(ws) or [])
    monkeypatch.setattr(app, "collection_has_documents", lambda ws: False)

    app._load_workspace_state("ws_a")
    app._load_workspace_state("ws_b")

    assert calls == ["ws_a", "ws_b"]


# ---------------------------------------------------------------------------
# _get_or_build_agent
# ---------------------------------------------------------------------------


def test_get_or_build_agent_builds_once_and_caches(monkeypatch):
    build_calls = []

    def _fake_build_agent(*, collection_name, metadata_filter):
        build_calls.append((collection_name, metadata_filter))
        return object()

    monkeypatch.setattr(app, "build_agent", _fake_build_agent)

    agent1 = app._get_or_build_agent("ws_x")
    agent2 = app._get_or_build_agent("ws_x")

    assert agent1 is agent2
    assert len(build_calls) == 1  # chỉ build 1 lần cho cùng 1 workspace


def test_get_or_build_agent_rebuilds_when_workspace_changes(monkeypatch):
    build_calls = []

    def _fake_build_agent(*, collection_name, metadata_filter):
        build_calls.append(collection_name)
        return f"agent-for-{collection_name}"

    monkeypatch.setattr(app, "build_agent", _fake_build_agent)

    agent1 = app._get_or_build_agent("ws_x")
    agent2 = app._get_or_build_agent("ws_y")

    assert agent1 != agent2
    assert build_calls == ["ws_x", "ws_y"]
