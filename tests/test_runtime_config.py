from __future__ import annotations

from pathlib import Path

import pytest

from src.runtime.config import ConfigError, RuntimeConfig, _read_env_file_value


def test_runtime_config_requires_schedule_source(monkeypatch):
    monkeypatch.delenv("TENNIS_SCHEDULE_SOURCE_URI", raising=False)
    monkeypatch.setenv("TENNIS_ODDS_SOURCE_URI", "/tmp/odds.csv")
    with pytest.raises(ConfigError, match="TENNIS_SCHEDULE_SOURCE_URI"):
        RuntimeConfig.from_env()


def test_runtime_config_allows_odds_api_without_odds_uri(tmp_path, monkeypatch):
    matches_path = tmp_path / "matches.parquet"
    rankings_path = tmp_path / "rankings.parquet"
    bundle_dir = tmp_path / "bundle"
    matches_path.write_bytes(b"stub")
    rankings_path.write_bytes(b"stub")
    bundle_dir.mkdir()

    monkeypatch.setenv("TENNIS_SOURCE_ADAPTER", "odds_api_tennis")
    monkeypatch.setenv("TENNIS_SCHEDULE_SOURCE_URI", "/tmp/schedule.csv")
    monkeypatch.delenv("TENNIS_ODDS_SOURCE_URI", raising=False)
    monkeypatch.setenv("ODDS_API_KEY", "test_key")
    monkeypatch.setenv("TENNIS_MATCH_DB_PATH", str(matches_path))
    monkeypatch.setenv("TENNIS_RANKINGS_PATH", str(rankings_path))
    monkeypatch.setenv("TENNIS_PROMOTED_BUNDLE_DIR", str(bundle_dir))

    cfg = RuntimeConfig.from_env()
    assert cfg.source_adapter == "odds_api_tennis"
    assert cfg.odds_source_uri is None
    assert cfg.odds_api_key == "test_key"


def test_read_env_file_value_strips_comments(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("ODDS_API_KEY=secret_value   # comment\n")
    assert _read_env_file_value(Path(env_path), "ODDS_API_KEY") == "secret_value"
