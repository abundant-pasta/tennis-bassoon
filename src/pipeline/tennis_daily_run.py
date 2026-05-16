"""
Shadow-mode daily orchestration for the tennis pipeline.

The production contract is intentionally fail-closed:
  1. fetch raw schedule + odds snapshots
  2. persist them immutably under a dated run folder
  3. validate schema, freshness, and promoted model bundle integrity
  4. build point-in-time features for the run date only
  5. score matches using the promoted bundle
  6. attach odds with explicit confidence thresholds
  7. emit predictions, recommendations, review artifacts, and a manifest

This job never executes wagers. It is shadow mode only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import argparse
import json
import os
import re
import urllib.request

import numpy as np
import pandas as pd

import config_tennis as cfg
from src.data.tennis_odds_api import fetch_tennis_odds_consensus
from src.data.tennis_sackmann import load_players
from src.features.tennis_engineer import build_features, get_feature_columns
from src.model.tennis_bundle import load_bundle, validate_bundle
from src.model.tennis_oos_report import _score_picks
from src.pipeline.tennis_odds_matcher import match_schedule_to_odds
from src.pipeline.tennis_shadow_ledger import (
    attach_line_movement,
    ensure_opening_odds_snapshot,
    finalize_close_snapshot,
    record_shadow_ledger,
    save_close_odds_snapshot,
    save_current_odds_snapshot,
)
from src.runtime.config import ConfigError, RuntimeConfig
from src.runtime.logging import log_event
from src.runtime.notifications import post_webhook
from src.runtime.storage import (
    ensure_dir,
    upload_directory_to_gcs,
    write_bytes,
    write_csv,
    write_json,
    write_parquet,
)


class DailyRunError(RuntimeError):
    """Raised when the daily run must fail closed."""


@dataclass(frozen=True)
class Snapshot:
    source_uri: str
    filename: str
    format: str
    fetched_utc: datetime
    source_updated_utc: datetime | None
    sha256_hex: str
    payload: bytes


@dataclass(frozen=True)
class PlayerRecord:
    player_id: int
    player_name: str
    player_hand: str | float
    player_ht: float
    player_age: float


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_name(text: str) -> str:
    return re.sub(r"[^a-z]", "", str(text).lower())


def _parse_http_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _source_filename(uri: str) -> str:
    if "://" in uri:
        tail = uri.rstrip("/").split("/")[-1]
    else:
        tail = Path(uri).name
    return tail or "snapshot.csv"


def _infer_format(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".csv", ".txt"}:
        return "csv"
    if suffix == ".json":
        return "json"
    if suffix in {".xlsx", ".xls"}:
        return "excel"
    raise DailyRunError(f"Unsupported snapshot format for {filename}")


def _fetch_snapshot(uri: str) -> Snapshot:
    fetched_utc = _utc_now()
    if uri.startswith(("http://", "https://")):
        req = urllib.request.Request(uri, headers={"User-Agent": "tennis-shadow-runner/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = resp.read()
            source_updated = _parse_http_datetime(resp.headers.get("Last-Modified"))
    else:
        path = Path(uri.replace("file://", "")).expanduser()
        payload = path.read_bytes()
        stat = path.stat()
        source_updated = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    filename = _source_filename(uri)
    return Snapshot(
        source_uri=uri,
        filename=filename,
        format=_infer_format(filename),
        fetched_utc=fetched_utc,
        source_updated_utc=source_updated,
        sha256_hex=sha256(payload).hexdigest(),
        payload=payload,
    )


def _load_snapshot_frame(snapshot: Snapshot) -> pd.DataFrame:
    if snapshot.format == "csv":
        return pd.read_csv(BytesIO(snapshot.payload))
    if snapshot.format == "json":
        return pd.DataFrame(json.loads(snapshot.payload.decode("utf-8")))
    if snapshot.format == "excel":
        return pd.read_excel(BytesIO(snapshot.payload))
    raise DailyRunError(f"Unsupported format {snapshot.format}")


def _fetch_odds_api_snapshot(runtime: RuntimeConfig, schedule_df: pd.DataFrame) -> tuple[Snapshot, pd.DataFrame, dict]:
    fetched_utc = _utc_now()
    odds_df, meta = fetch_tennis_odds_consensus(
        api_key=runtime.odds_api_key or "",
        schedule_df=schedule_df,
        api_base=runtime.odds_api_base,
        regions=runtime.odds_api_regions,
        bookmakers=runtime.odds_api_bookmakers,
        markets=runtime.odds_api_markets,
        odds_format=runtime.odds_api_odds_format,
        override_sport_keys=runtime.odds_api_sport_keys,
    )
    raw_payload = meta["raw_payload"]
    snapshot = Snapshot(
        source_uri="the-odds-api",
        filename="raw_odds.json",
        format="json",
        fetched_utc=fetched_utc,
        source_updated_utc=fetched_utc,
        sha256_hex=sha256(raw_payload).hexdigest(),
        payload=raw_payload,
    )
    return snapshot, odds_df, meta


def _validate_snapshot_freshness(snapshot: Snapshot, max_age_hours: int, label: str) -> dict:
    if snapshot.source_updated_utc is None:
        raise DailyRunError(f"{label} snapshot is missing source_updated_utc metadata.")
    age_hours = (_utc_now() - snapshot.source_updated_utc).total_seconds() / 3600.0
    if age_hours > max_age_hours:
        raise DailyRunError(
            f"{label} snapshot is stale: {age_hours:.2f}h old > {max_age_hours}h threshold."
        )
    return {
        "source_uri": snapshot.source_uri,
        "filename": snapshot.filename,
        "sha256": snapshot.sha256_hex,
        "fetched_utc": snapshot.fetched_utc.isoformat(),
        "source_updated_utc": snapshot.source_updated_utc.isoformat(),
        "age_hours": round(age_hours, 2),
    }


def _require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DailyRunError(f"{label} snapshot missing required columns: {missing}")


def _normalize_schedule(df: pd.DataFrame, run_date: pd.Timestamp) -> pd.DataFrame:
    required = ["match_date", "tourney_name", "surface", "round", "player_name", "opp_name"]
    _require_columns(df, required, "schedule")
    out = df.copy()
    out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce")
    out = out[out["match_date"].notna()].copy()
    out = out[out["match_date"].dt.date == run_date.date()].copy()
    if out.empty:
        raise DailyRunError(f"No schedule rows found for {run_date.date()}.")
    out["best_of"] = pd.to_numeric(out.get("best_of", 3), errors="coerce").fillna(3).astype(int)
    out["draw_size"] = pd.to_numeric(out.get("draw_size", 32), errors="coerce").fillna(32).astype(int)
    out["player_rank"] = pd.to_numeric(out.get("player_rank"), errors="coerce")
    out["opp_rank"] = pd.to_numeric(out.get("opp_rank"), errors="coerce")
    out["match_id"] = out.get("match_id", "")
    missing_ids = out["match_id"].isna() | (out["match_id"].astype(str).str.strip() == "")
    out.loc[missing_ids, "match_id"] = out.loc[missing_ids].apply(
        lambda r: (
            f"daily_{pd.Timestamp(r['match_date']).date()}_"
            f"{_normalize_name(r['tourney_name'])}_{_normalize_name(r['player_name'])}_{_normalize_name(r['opp_name'])}"
        ),
        axis=1,
    )
    return _canonicalize_schedule_side(out)


def _normalize_odds(df: pd.DataFrame, run_date: pd.Timestamp) -> pd.DataFrame:
    required = [
        "match_date",
        "tourney_name",
        "surface",
        "round",
        "player_name",
        "opp_name",
        "player_decimal_odds",
        "opp_decimal_odds",
    ]
    _require_columns(df, required, "odds")
    out = df.copy()
    out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce", utc=True)
    out = out[out["match_date"].notna()].copy()
    run_start = pd.Timestamp(run_date.date(), tz="UTC")
    out = out[
        (out["match_date"] >= run_start - pd.Timedelta(days=1))
        & (out["match_date"] < run_start + pd.Timedelta(days=2))
    ].copy()
    if out.empty:
        raise DailyRunError(f"No odds rows found for {run_date.date()}.")
    out["player_decimal_odds"] = pd.to_numeric(out["player_decimal_odds"], errors="coerce")
    out["opp_decimal_odds"] = pd.to_numeric(out["opp_decimal_odds"], errors="coerce")
    out = out[
        out["player_decimal_odds"].notna()
        & out["opp_decimal_odds"].notna()
        & (out["player_decimal_odds"] > 1.0)
        & (out["opp_decimal_odds"] > 1.0)
    ].copy()
    if out.empty:
        raise DailyRunError("No valid decimal odds remained after normalization.")
    if "provider" not in out.columns:
        out["provider"] = "generic_csv"
    return _canonicalize_schedule_side(out)


def _canonicalize_schedule_side(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _should_swap(row: pd.Series) -> bool:
        pr = pd.to_numeric(row.get("player_rank"), errors="coerce")
        orank = pd.to_numeric(row.get("opp_rank"), errors="coerce")
        if pd.notna(pr) and pd.notna(orank):
            if pr > orank:
                return True
            if pr < orank:
                return False
        elif pd.isna(pr) and pd.notna(orank):
            return True
        elif pd.notna(pr) and pd.isna(orank):
            return False
        return str(row.get("player_name", "")) > str(row.get("opp_name", ""))

    for idx, row in out.iterrows():
        if not _should_swap(row):
            continue
        swap_pairs = [
            ("player_name", "opp_name"),
            ("player_rank", "opp_rank"),
            ("player_rank_points", "opp_rank_points"),
            ("player_id", "opp_id"),
            ("player_hand", "opp_hand"),
            ("player_ht", "opp_ht"),
            ("player_age", "opp_age"),
            ("player_decimal_odds", "opp_decimal_odds"),
        ]
        for left, right in swap_pairs:
            if left in out.columns and right in out.columns:
                out.at[idx, left], out.at[idx, right] = out.at[idx, right], out.at[idx, left]
    return out.reset_index(drop=True)


def _build_player_lookup(players_master: pd.DataFrame) -> dict[str, list[dict]]:
    master = players_master.copy()
    if master.empty:
        return {}
    master["surname_key"] = master["name_last"].map(_normalize_name)
    master["first_key"] = master["name_first"].map(_normalize_name)
    master["initials_key"] = master["name_first"].map(
        lambda s: "".join(p[0] for p in re.split(r"[^A-Za-z]+", str(s).lower()) if p)
    )
    master["player_name"] = master["name_first"].fillna("").astype(str).str.strip() + " " + master["name_last"].fillna("").astype(str).str.strip()
    lookup: dict[str, list[dict]] = {}
    for rec in master.to_dict("records"):
        lookup.setdefault(rec["surname_key"], []).append(rec)
    return lookup


def _resolve_player(name: str, match_date: pd.Timestamp, lookup: dict[str, list[dict]], synthetic_ids: dict[str, int]) -> PlayerRecord:
    cleaned = str(name).strip().replace(".", "")
    parts = cleaned.split()

    # Try "First Last" format (Odds API) — surname is the last token
    # Fall back to "Last Initial" format — surname is the first token(s)
    if len(parts) >= 2:
        # "First Last": surname = last word, token = first word
        surname_key_fl = _normalize_name(parts[-1])
        token_fl = _normalize_name(parts[0])
        # "Last Initial": surname = all-but-last, token = last word
        surname_key_li = _normalize_name(" ".join(parts[:-1]))
        token_li = _normalize_name(parts[-1])
    else:
        surname_key_fl = surname_key_li = _normalize_name(cleaned)
        token_fl = token_li = ""

    candidates = lookup.get(surname_key_fl, []) or lookup.get(surname_key_li, [])
    surname_key = surname_key_fl if lookup.get(surname_key_fl) else surname_key_li
    token = token_fl if lookup.get(surname_key_fl) else token_li
    if candidates:
        filtered = [
            rec for rec in candidates
            if str(rec["initials_key"]).startswith(token) or str(rec["first_key"]).startswith(token)
        ]
        if not filtered:
            filtered = candidates
        chosen = filtered[0]
        dob = pd.to_datetime(str(int(chosen["dob"])) if pd.notna(chosen.get("dob")) else "", format="%Y%m%d", errors="coerce")
        match_date_naive = match_date.tz_localize(None) if getattr(match_date, "tzinfo", None) is not None else match_date
        age = ((match_date_naive - dob).days / 365.25) if pd.notna(dob) else np.nan
        return PlayerRecord(
            player_id=int(chosen["player_id"]),
            player_name=chosen["player_name"].strip(),
            player_hand=chosen.get("hand", np.nan),
            player_ht=float(chosen["height"]) if pd.notna(chosen.get("height")) else np.nan,
            player_age=age,
        )
    key = _normalize_name(cleaned)
    if key not in synthetic_ids:
        synthetic_ids[key] = 800_000_000 + len(synthetic_ids) + 1
    return PlayerRecord(
        player_id=synthetic_ids[key],
        player_name=cleaned,
        player_hand=np.nan,
        player_ht=np.nan,
        player_age=np.nan,
    )


def _latest_rank_lookup(rankings: pd.DataFrame, run_date: pd.Timestamp) -> dict[int, tuple[float, float]]:
    run_date_naive = run_date.tz_localize(None) if run_date.tzinfo is not None else run_date
    hist = rankings[pd.to_datetime(rankings["ranking_date"]) <= run_date_naive].copy()
    hist = hist.sort_values(["player_id", "ranking_date"]).drop_duplicates("player_id", keep="last")
    return {
        int(row["player_id"]): (float(row["rank"]), float(row.get("points", np.nan)))
        for _, row in hist.iterrows()
        if pd.notna(row["player_id"]) and pd.notna(row["rank"])
    }


def _schedule_to_player_rows(schedule: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    players_master = load_players(verbose=False)
    lookup = _build_player_lookup(players_master)
    rank_lookup = _latest_rank_lookup(rankings, pd.Timestamp(schedule["match_date"].iloc[0]))
    synthetic_ids: dict[str, int] = {}
    rows: list[dict] = []

    for row in schedule.to_dict("records"):
        match_date = pd.Timestamp(row["match_date"])
        player = _resolve_player(row["player_name"], match_date, lookup, synthetic_ids)
        opp = _resolve_player(row["opp_name"], match_date, lookup, synthetic_ids)
        player_rank, player_points = rank_lookup.get(player.player_id, (np.nan, np.nan))
        opp_rank, opp_points = rank_lookup.get(opp.player_id, (np.nan, np.nan))
        base = {
            "match_id": row["match_id"],
            "tourney_date": match_date,
            "sim_date": match_date,
            "tourney_name": row["tourney_name"],
            "surface": row["surface"],
            "tourney_level": row.get("tourney_level", "A"),
            "draw_size": row.get("draw_size", 32),
            "round": row["round"],
            "best_of": row.get("best_of", 3),
            "score": "",
            "minutes": np.nan,
            "is_walkover": False,
            "is_retirement": False,
            "data_source": "daily_schedule",
            "won": np.nan,
        }
        player_rank_value = row.get("player_rank", np.nan)
        opp_rank_value = row.get("opp_rank", np.nan)
        rows.append({
            **base,
            "player_id": player.player_id,
            "player_name": player.player_name,
            "player_hand": row.get("player_hand", player.player_hand),
            "player_ht": row.get("player_ht", player.player_ht),
            "player_age": row.get("player_age", player.player_age),
            "player_rank": player_rank_value if pd.notna(player_rank_value) else player_rank,
            "player_rank_points": row.get("player_rank_points", player_points),
            "opp_id": opp.player_id,
            "opp_name": opp.player_name,
            "opp_hand": row.get("opp_hand", opp.player_hand),
            "opp_ht": row.get("opp_ht", opp.player_ht),
            "opp_age": row.get("opp_age", opp.player_age),
            "opp_rank": opp_rank_value if pd.notna(opp_rank_value) else opp_rank,
            "opp_rank_points": row.get("opp_rank_points", opp_points),
            "ace": np.nan,
            "df": np.nan,
            "svpt": np.nan,
            "first_in": np.nan,
            "first_won": np.nan,
            "second_won": np.nan,
            "sv_gms": np.nan,
            "bp_saved": np.nan,
            "bp_faced": np.nan,
            "opp_ace": np.nan,
            "opp_df": np.nan,
            "opp_svpt": np.nan,
            "opp_first_in": np.nan,
            "opp_first_won": np.nan,
            "opp_second_won": np.nan,
            "opp_sv_gms": np.nan,
            "opp_bp_saved": np.nan,
            "opp_bp_faced": np.nan,
        })
    return pd.DataFrame(rows)


def _build_run_dir(run_root: Path, run_date: pd.Timestamp) -> tuple[str, Path]:
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S%fZ")
    run_dir = run_root / run_date.strftime("%Y") / run_date.strftime("%m") / run_date.strftime("%d") / run_id
    ensure_dir(run_dir)
    return run_id, run_dir


def _git_sha() -> str | None:
    head = Path(".git/HEAD")
    if not head.exists():
        return None
    text = head.read_text().strip()
    if text.startswith("ref: "):
        ref_path = Path(".git") / text.replace("ref: ", "")
        if ref_path.exists():
            return ref_path.read_text().strip()
    return text or None


def _write_manifest(run_dir: Path, manifest: dict) -> Path:
    return write_json(run_dir / "run_manifest.json", manifest)


def run_daily(run_date: str, mode: str = "shadow") -> dict:
    if mode not in {"shadow", "close_snapshot"}:
        raise DailyRunError("Supported modes are 'shadow' and 'close_snapshot'.")

    run_dt = pd.Timestamp(run_date).normalize()
    fallback_run_root = Path(os.getenv("TENNIS_RUN_ROOT", str(cfg.RUNS_DIR))).expanduser()
    run_id, run_dir = _build_run_dir(fallback_run_root, run_dt)
    runtime: RuntimeConfig | None = None
    validation_results: dict[str, object] = {}
    manifest = {
        "run_id": run_id,
        "run_date": str(run_dt.date()),
        "mode": mode,
        "env": os.getenv("TENNIS_ENV", "dev"),
        "git_sha": _git_sha(),
        "status": "started",
        "created_utc": _utc_now().isoformat(),
        "validation_results": validation_results,
    }
    _write_manifest(run_dir, manifest)

    try:
        runtime = RuntimeConfig.from_env()
        if runtime.run_root != fallback_run_root:
            run_dir = runtime.run_root / run_dt.strftime("%Y") / run_dt.strftime("%m") / run_dt.strftime("%d") / run_id
            ensure_dir(run_dir)
            _write_manifest(run_dir, manifest)
        manifest["env"] = runtime.env
        log_event(
            "daily_run",
            "starting_shadow_run" if mode == "shadow" else "starting_close_snapshot",
            run_id=run_id,
            run_date=str(run_dt.date()),
        )
        schedule_snapshot = _fetch_snapshot(runtime.schedule_source_uri)
        validation_results["schedule_snapshot"] = _validate_snapshot_freshness(
            schedule_snapshot, runtime.schedule_max_age_hours, "schedule"
        )
        raw_schedule_path = write_bytes(run_dir / f"raw_schedule{Path(schedule_snapshot.filename).suffix}", schedule_snapshot.payload)
        schedule_df = _normalize_schedule(_load_snapshot_frame(schedule_snapshot), run_dt)
        odds_meta: dict[str, object] = {}
        if runtime.source_adapter == "odds_api_tennis":
            odds_snapshot, odds_df_raw, odds_meta = _fetch_odds_api_snapshot(runtime, schedule_df)
        else:
            odds_snapshot = _fetch_snapshot(runtime.odds_source_uri or "")
            odds_df_raw = _load_snapshot_frame(odds_snapshot)
        validation_results["odds_snapshot"] = _validate_snapshot_freshness(
            odds_snapshot, runtime.odds_max_age_hours, "odds"
        )
        if odds_meta:
            validation_results["odds_api"] = {
                "requested_tournaments": odds_meta.get("requested_tournaments", []),
                "unsupported_tournaments": odds_meta.get("unsupported_tournaments", []),
                "request_headers": odds_meta.get("request_headers", []),
            }
        raw_odds_path = write_bytes(run_dir / f"raw_odds{Path(odds_snapshot.filename).suffix}", odds_snapshot.payload)
        odds_df = _normalize_odds(odds_df_raw, run_dt)
        match_result = match_schedule_to_odds(schedule_df, odds_df)
        matched_odds = match_result.matched.copy()
        rejected_odds = match_result.rejected.copy()
        if matched_odds.empty:
            raise DailyRunError("No schedule rows passed odds matching.")

        if mode == "close_snapshot":
            close_snapshot_path = save_close_odds_snapshot(str(run_dt.date()), run_id, matched_odds)
            close_update = finalize_close_snapshot(str(run_dt.date()), close_snapshot_path)
            write_csv(run_dir / "odds_review.csv", rejected_odds)

            manifest.update(
                {
                    "status": "success",
                    "completed_utc": _utc_now().isoformat(),
                    "row_counts": {
                        "schedule_rows": int(len(schedule_df)),
                        "odds_rows": int(len(odds_df)),
                        "matched_odds_rows": int(len(matched_odds)),
                        "odds_review_rows": int(len(rejected_odds)),
                        "shadow_ledger_target_rows": int(close_update["target_rows"]),
                        "shadow_ledger_close_rows": int(close_update["matched_close_rows"]),
                        "shadow_ledger_beat_close_rows": int(close_update["beat_close_rows"]),
                    },
                    "artifacts": {
                        "raw_schedule": str(raw_schedule_path),
                        "raw_odds": str(raw_odds_path),
                        "close_odds_snapshot": str(close_snapshot_path),
                        "odds_review": str(run_dir / "odds_review.csv"),
                        "shadow_ledger": str(cfg.SHADOW_LEDGER_PATH),
                    },
                }
            )
        else:
            rankings = pd.read_parquet(runtime.rankings_path)
            historical_matches = pd.read_parquet(runtime.matches_path)
            if "sim_date" not in historical_matches.columns:
                historical_matches["sim_date"] = historical_matches["tourney_date"]
            historical_matches = historical_matches[pd.to_datetime(historical_matches["sim_date"]) < run_dt].copy()
            if historical_matches.empty:
                raise DailyRunError("Historical match DB had no rows before the requested run date.")

            bundle_info = load_bundle(runtime.promoted_bundle_dir)
            validate_bundle(runtime.promoted_bundle_dir)
            validation_results["bundle_version"] = bundle_info["manifest"]["version"]
            validation_results["feature_checksum"] = bundle_info["manifest"]["feature_checksum"]

            schedule_rows = _schedule_to_player_rows(schedule_df, rankings)
            combined = pd.concat([historical_matches, schedule_rows], ignore_index=True, sort=False)
            features = build_features(combined, verbose=False)
            feature_rows = features[features["data_source"] == "daily_schedule"].copy()
            if feature_rows.empty:
                raise DailyRunError("No schedule feature rows were generated for the run date.")

            feat_cols = get_feature_columns()
            feature_rows["feature_coverage"] = feature_rows[feat_cols].notna().mean(axis=1)
            min_coverage = float(feature_rows["feature_coverage"].min())
            validation_results["feature_coverage_min"] = round(min_coverage, 4)
            validation_results["feature_coverage_mean"] = round(float(feature_rows["feature_coverage"].mean()), 4)
            if min_coverage < runtime.min_feature_coverage:
                raise DailyRunError(
                    f"Feature coverage dropped below threshold: {min_coverage:.4f} < {runtime.min_feature_coverage:.4f}"
                )

            feature_rows[feat_cols] = feature_rows[feat_cols].fillna(pd.Series(bundle_info["medians"]))
            X = bundle_info["scaler"].transform(feature_rows[feat_cols].values)
            feature_rows["player_model_prob_raw"] = bundle_info["model"].predict_proba(X)[:, 1]
            feature_rows["player_model_prob"] = feature_rows["player_model_prob_raw"]
            feature_rows["opp_model_prob_raw"] = 1.0 - feature_rows["player_model_prob_raw"]
            feature_rows["opp_model_prob"] = 1.0 - feature_rows["player_model_prob"]

            current_snapshot_path = save_current_odds_snapshot(str(run_dt.date()), run_id, matched_odds)
            opening_snapshot_path = ensure_opening_odds_snapshot(str(run_dt.date()), matched_odds)

            predictions = feature_rows.merge(
                matched_odds[
                    [
                        "match_id",
                        "player_decimal_odds",
                        "opp_decimal_odds",
                        "novig_player_prob",
                        "novig_opp_prob",
                        "odds_match_confidence",
                        "odds_provider",
                        "date_distance_days",
                    ]
                ],
                on="match_id",
                how="inner",
            )
            if predictions.empty:
                raise DailyRunError("No prediction rows remained after odds matching.")

            predictions["player_market_prob"] = predictions["novig_player_prob"]
            predictions["opp_market_prob"] = predictions["novig_opp_prob"]
            predictions["won"] = np.nan
            scored = _score_picks(predictions, edge_threshold=cfg.EDGE_THRESHOLD)
            low_conf_mask = scored["odds_match_confidence"] < runtime.min_odds_match_confidence
            scored["low_confidence_block"] = np.where(
                low_conf_mask,
                "odds_match_confidence_below_threshold",
                None,
            )
            scored["eligible"] = scored["eligible"] & (~low_conf_mask)
            scored["shadow_mode"] = True
            scored = attach_line_movement(scored, opening_snapshot_path)

            recommendations = scored[scored["eligible"]].copy()
            ledger_rows = record_shadow_ledger(str(run_dt.date()), run_id, scored)
            write_parquet(run_dir / "features.parquet", feature_rows)
            write_csv(run_dir / "predictions.csv", scored)
            write_csv(run_dir / "recommendations.csv", recommendations)
            write_csv(run_dir / "odds_review.csv", rejected_odds)

            manifest.update(
                {
                    "status": "success",
                    "completed_utc": _utc_now().isoformat(),
                    "model_artifact_version": bundle_info["manifest"]["version"],
                    "row_counts": {
                        "schedule_rows": int(len(schedule_df)),
                        "odds_rows": int(len(odds_df)),
                        "historical_rows": int(len(historical_matches)),
                        "feature_rows": int(len(feature_rows)),
                        "matched_odds_rows": int(len(matched_odds)),
                        "odds_review_rows": int(len(rejected_odds)),
                        "prediction_rows": int(len(scored)),
                        "recommendation_rows": int(len(recommendations)),
                        "shadow_ledger_rows": int(ledger_rows),
                    },
                    "artifacts": {
                        "raw_schedule": str(raw_schedule_path),
                        "raw_odds": str(raw_odds_path),
                        "features": str(run_dir / "features.parquet"),
                        "predictions": str(run_dir / "predictions.csv"),
                        "recommendations": str(run_dir / "recommendations.csv"),
                        "odds_review": str(run_dir / "odds_review.csv"),
                        "opening_odds_snapshot": str(opening_snapshot_path),
                        "current_odds_snapshot": str(current_snapshot_path),
                        "shadow_ledger": str(cfg.SHADOW_LEDGER_PATH),
                    },
                }
            )
        if runtime.upload_to_gcs and runtime.output_bucket:
            prefix = str(run_dir.relative_to(runtime.run_root))
            uploaded = upload_directory_to_gcs(run_dir, runtime.output_bucket, prefix)
            manifest["gcs_artifacts"] = uploaded

        _write_manifest(run_dir, manifest)
        log_event(
            "daily_run",
            "shadow_run_complete" if mode == "shadow" else "close_snapshot_complete",
            run_id=run_id,
            recommendation_rows=int(manifest.get("row_counts", {}).get("recommendation_rows", 0)),
            odds_review_rows=int(len(rejected_odds)),
        )
        if runtime.notification_webhook:
            post_webhook(
                runtime.notification_webhook,
                {
                    "run_id": run_id,
                    "run_date": str(run_dt.date()),
                    "mode": mode,
                    "status": manifest["status"],
                    "recommendation_rows": int(manifest.get("row_counts", {}).get("recommendation_rows", 0)),
                    "odds_review_rows": int(len(rejected_odds)),
                },
            )
        return manifest
    except (ConfigError, DailyRunError, FileNotFoundError, ValueError) as exc:
        manifest["status"] = "failed"
        manifest["completed_utc"] = _utc_now().isoformat()
        manifest["error"] = str(exc)
        _write_manifest(run_dir, manifest)
        log_event("daily_run", "shadow_run_failed", run_id=run_id, error=str(exc))
        if runtime and runtime.notification_webhook:
            post_webhook(
                runtime.notification_webhook,
                {"run_id": run_id, "run_date": str(run_dt.date()), "status": "failed", "error": str(exc)},
            )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the tennis shadow-mode daily pipeline.")
    parser.add_argument("--run-date", required=True, help="Run date in YYYY-MM-DD format.")
    parser.add_argument("--mode", default="shadow", choices=["shadow", "close_snapshot"])
    args = parser.parse_args()

    summary = run_daily(run_date=args.run_date, mode=args.mode)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
