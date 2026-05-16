"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import config_tennis as cfg


class ConfigError(ValueError):
    """Raised when runtime configuration is invalid."""


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    if value is None:
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    if value is None:
        return default
    return int(value)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_env_file_value(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        left, right = line.split("=", 1)
        if left.strip() != key:
            continue
        return right.split("#", 1)[0].strip().strip("'\"")
    return None


def _resolve_odds_api_key() -> str | None:
    for env_name in ("TENNIS_ODDS_API_KEY", "ODDS_API_KEY"):
        value = _env(env_name)
        if value:
            return value
    sibling_repo = _repo_root().parent / "laughing-bassoon"
    for path in (sibling_repo / ".env", sibling_repo / "remote_config.env"):
        value = _read_env_file_value(path, "ODDS_API_KEY")
        if value:
            return value
    return None


@dataclass(frozen=True)
class RuntimeConfig:
    env: str
    run_root: Path
    output_bucket: str | None
    notification_webhook: str | None
    source_adapter: str
    schedule_source_uri: str
    odds_source_uri: str | None
    schedule_max_age_hours: int
    odds_max_age_hours: int
    min_feature_coverage: float
    min_odds_match_confidence: float
    matches_path: Path
    rankings_path: Path
    promoted_bundle_dir: Path
    timezone: str
    upload_to_gcs: bool
    odds_api_key: str | None
    odds_api_base: str
    odds_api_regions: str
    odds_api_bookmakers: str | None
    odds_api_markets: str
    odds_api_odds_format: str
    odds_api_sport_keys: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        env = _env("TENNIS_ENV", "dev")
        source_adapter = _env("TENNIS_SOURCE_ADAPTER", "generic_csv")
        schedule_source_uri = _env("TENNIS_SCHEDULE_SOURCE_URI")
        odds_source_uri = _env("TENNIS_ODDS_SOURCE_URI")
        if not schedule_source_uri:
            raise ConfigError("TENNIS_SCHEDULE_SOURCE_URI is required.")
        if source_adapter == "generic_csv" and not odds_source_uri:
            raise ConfigError("TENNIS_ODDS_SOURCE_URI is required.")

        run_root = Path(_env("TENNIS_RUN_ROOT", str(cfg.RUNS_DIR))).expanduser()
        output_bucket = _env("TENNIS_OUTPUT_BUCKET")
        upload_to_gcs = _env_bool("TENNIS_UPLOAD_TO_GCS", bool(output_bucket))
        promoted_bundle_dir = Path(
            _env("TENNIS_PROMOTED_BUNDLE_DIR", str(cfg.PROMOTED_CURRENT_DIR))
        ).expanduser()
        default_matches_path = cfg.LEDGER_DIR / "extended_matches.parquet"
        if not default_matches_path.exists():
            default_matches_path = cfg.FEATURES_DIR / "matches.parquet"
        matches_path = Path(_env("TENNIS_MATCH_DB_PATH", str(default_matches_path))).expanduser()
        rankings_path = Path(_env("TENNIS_RANKINGS_PATH", str(cfg.FEATURES_DIR / "rankings.parquet"))).expanduser()

        runtime = cls(
            env=env,
            run_root=run_root,
            output_bucket=output_bucket,
            notification_webhook=_env("TENNIS_NOTIFICATION_WEBHOOK"),
            source_adapter=source_adapter,
            schedule_source_uri=schedule_source_uri,
            odds_source_uri=odds_source_uri,
            schedule_max_age_hours=_env_int("TENNIS_SCHEDULE_MAX_AGE_HOURS", cfg.MAX_SNAPSHOT_AGE_HOURS),
            odds_max_age_hours=_env_int("TENNIS_ODDS_MAX_AGE_HOURS", cfg.MAX_SNAPSHOT_AGE_HOURS),
            min_feature_coverage=_env_float("TENNIS_MIN_FEATURE_COVERAGE", cfg.MIN_FEATURE_COVERAGE),
            min_odds_match_confidence=_env_float(
                "TENNIS_MIN_ODDS_MATCH_CONFIDENCE", cfg.MIN_ODDS_MATCH_CONFIDENCE
            ),
            matches_path=matches_path,
            rankings_path=rankings_path,
            promoted_bundle_dir=promoted_bundle_dir,
            timezone=_env("TENNIS_TIMEZONE", cfg.PRODUCTION_TIMEZONE),
            upload_to_gcs=upload_to_gcs,
            odds_api_key=_resolve_odds_api_key(),
            odds_api_base=_env("TENNIS_ODDS_API_BASE", "https://api.the-odds-api.com/v4"),
            odds_api_regions=_env("TENNIS_ODDS_API_REGIONS", "us"),
            odds_api_bookmakers=_env("TENNIS_ODDS_API_BOOKMAKERS"),
            odds_api_markets=_env("TENNIS_ODDS_API_MARKETS", "h2h"),
            odds_api_odds_format=_env("TENNIS_ODDS_API_ODDS_FORMAT", "decimal"),
            odds_api_sport_keys=tuple(
                key.strip()
                for key in (_env("TENNIS_ODDS_API_SPORT_KEYS", "") or "").split(",")
                if key.strip()
            ),
        )
        runtime.validate()
        return runtime

    def validate(self) -> None:
        if self.source_adapter not in {"generic_csv", "odds_api_tennis"}:
            raise ConfigError(
                f"Unsupported TENNIS_SOURCE_ADAPTER={self.source_adapter!r}. "
                "Supported adapters are 'generic_csv' and 'odds_api_tennis'."
            )
        if self.source_adapter == "odds_api_tennis":
            if not self.odds_api_key:
                raise ConfigError(
                    "ODDS_API_KEY or TENNIS_ODDS_API_KEY is required when "
                    "TENNIS_SOURCE_ADAPTER=odds_api_tennis."
                )
            if self.odds_api_markets != "h2h":
                raise ConfigError("TENNIS_ODDS_API_MARKETS currently only supports 'h2h'.")
            if self.odds_api_odds_format != "decimal":
                raise ConfigError("TENNIS_ODDS_API_ODDS_FORMAT currently must be 'decimal'.")
        if self.schedule_max_age_hours <= 0:
            raise ConfigError("TENNIS_SCHEDULE_MAX_AGE_HOURS must be positive.")
        if self.odds_max_age_hours <= 0:
            raise ConfigError("TENNIS_ODDS_MAX_AGE_HOURS must be positive.")
        if not 0.0 <= self.min_feature_coverage <= 1.0:
            raise ConfigError("TENNIS_MIN_FEATURE_COVERAGE must be between 0 and 1.")
        if not 0.0 <= self.min_odds_match_confidence <= 1.0:
            raise ConfigError("TENNIS_MIN_ODDS_MATCH_CONFIDENCE must be between 0 and 1.")
        if not self.matches_path.exists():
            raise ConfigError(f"Historical match DB not found: {self.matches_path}")
        if not self.rankings_path.exists():
            raise ConfigError(f"Rankings DB not found: {self.rankings_path}")
        if not self.promoted_bundle_dir.exists():
            raise ConfigError(f"Promoted model bundle not found: {self.promoted_bundle_dir}")
        if self.upload_to_gcs and not self.output_bucket:
            raise ConfigError("TENNIS_OUTPUT_BUCKET is required when TENNIS_UPLOAD_TO_GCS=true.")
