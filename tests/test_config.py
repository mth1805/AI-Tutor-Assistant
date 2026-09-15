"""Test config.py: đọc biến môi trường, validate bắt buộc, giá trị mặc định."""

from __future__ import annotations

import pytest

from config import AppConfig, DatabaseConfig, _get_env


def test_get_env_required_missing_raises(monkeypatch):
    monkeypatch.delenv("SOME_UNSET_VAR", raising=False)
    with pytest.raises(EnvironmentError):
        _get_env("SOME_UNSET_VAR", required=True)


def test_get_env_uses_default_when_missing(monkeypatch):
    monkeypatch.delenv("SOME_UNSET_VAR", raising=False)
    assert _get_env("SOME_UNSET_VAR", "mac_dinh") == "mac_dinh"


def test_get_env_returns_actual_value_when_set(monkeypatch):
    monkeypatch.setenv("SOME_VAR", "gia_tri_that")
    assert _get_env("SOME_VAR", "mac_dinh") == "gia_tri_that"


def test_database_config_requires_user_and_password(monkeypatch):
    monkeypatch.delenv("PG_USER", raising=False)
    monkeypatch.delenv("PG_PASSWORD", raising=False)
    with pytest.raises(EnvironmentError):
        DatabaseConfig()


def test_database_config_connection_string(monkeypatch):
    monkeypatch.setenv("PG_USER", "u")
    monkeypatch.setenv("PG_PASSWORD", "p")
    monkeypatch.setenv("PG_HOST", "myhost")
    monkeypatch.setenv("PG_PORT", "5433")
    monkeypatch.setenv("PG_DATABASE", "db1")
    cfg = DatabaseConfig()
    assert cfg.connection_string == "postgresql+psycopg://u:p@myhost:5433/db1"


def test_app_config_defaults(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "g")
    monkeypatch.setenv("COHERE_API_KEY", "c")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("CHUNKING_STRATEGY", raising=False)
    monkeypatch.delenv("ENABLE_OCR", raising=False)
    cfg = AppConfig()
    assert cfg.embedding_model == "BAAI/bge-m3"
    assert cfg.chunking_strategy == "fixed"
    assert cfg.enable_ocr is True


def test_app_config_enable_ocr_false(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "g")
    monkeypatch.setenv("COHERE_API_KEY", "c")
    monkeypatch.setenv("ENABLE_OCR", "false")
    cfg = AppConfig()
    assert cfg.enable_ocr is False
