from __future__ import annotations

from hashlib import sha256
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

from src.features.tennis_engineer import get_feature_columns
from src.pipeline.tennis_daily_run import run_daily


def _historical_rows() -> pd.DataFrame:
    rows = []
    matches = [
        ("hist1", "2026-01-01", 1, "Alpha One", 2, "Beta Two", 1),
        ("hist2", "2026-01-02", 2, "Beta Two", 1, "Alpha One", 0),
        ("hist3", "2026-01-03", 1, "Alpha One", 3, "Gamma Three", 1),
        ("hist4", "2026-01-04", 3, "Gamma Three", 1, "Alpha One", 0),
    ]
    for match_id, date, pid, pname, oid, oname, won in matches:
        rows.append(
            {
                "match_id": match_id,
                "tourney_date": pd.Timestamp(date),
                "sim_date": pd.Timestamp(date),
                "tourney_name": "Test Event",
                "surface": "Hard",
                "tourney_level": "A",
                "draw_size": 32,
                "round": "R32",
                "best_of": 3,
                "score": "6-4 6-4",
                "minutes": 90,
                "is_walkover": False,
                "is_retirement": False,
                "data_source": "test_fixture",
                "player_id": pid,
                "player_name": pname,
                "player_hand": "R",
                "player_ht": 185,
                "player_age": 25.0,
                "player_rank": 10 if pid == 1 else 20 if pid == 2 else 30,
                "player_rank_points": 1000,
                "opp_id": oid,
                "opp_name": oname,
                "opp_hand": "R",
                "opp_ht": 188,
                "opp_age": 26.0,
                "opp_rank": 10 if oid == 1 else 20 if oid == 2 else 30,
                "opp_rank_points": 900,
                "won": won,
                "ace": 5,
                "df": 2,
                "svpt": 60,
                "first_in": 35,
                "first_won": 28,
                "second_won": 12,
                "sv_gms": 10,
                "bp_saved": 3,
                "bp_faced": 5,
                "opp_ace": 4,
                "opp_df": 3,
                "opp_svpt": 58,
                "opp_first_in": 32,
                "opp_first_won": 22,
                "opp_second_won": 10,
                "opp_sv_gms": 10,
                "opp_bp_saved": 2,
                "opp_bp_faced": 5,
            }
        )
    return pd.DataFrame(rows)


def test_daily_run_writes_shadow_artifacts(tmp_path, monkeypatch):
    feature_cols = get_feature_columns()
    bundle_dir = tmp_path / "promoted" / "current"
    bundle_dir.mkdir(parents=True)

    model = DummyClassifier(strategy="prior")
    X_fit = np.zeros((2, len(feature_cols)))
    y_fit = np.array([0, 1])
    model.fit(X_fit, y_fit)
    scaler = StandardScaler().fit(X_fit)
    medians = {col: 0.0 for col in feature_cols}

    joblib.dump(model, bundle_dir / "model.pkl")
    joblib.dump(scaler, bundle_dir / "scaler.pkl")
    joblib.dump(medians, bundle_dir / "medians.pkl")
    (bundle_dir / "metadata.json").write_text(json.dumps({"val_auc_roc": 0.75}))
    (bundle_dir / "feature_columns.json").write_text(json.dumps(feature_cols))
    (bundle_dir / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "version": "test_bundle",
                "feature_checksum": sha256("\n".join(feature_cols).encode("utf-8")).hexdigest(),
            }
        )
    )

    matches_path = tmp_path / "matches.parquet"
    rankings_path = tmp_path / "rankings.parquet"
    _historical_rows().to_parquet(matches_path, index=False)
    pd.DataFrame(
        [
            {"player_id": 1, "ranking_date": pd.Timestamp("2026-01-05"), "rank": 10, "points": 1000},
            {"player_id": 2, "ranking_date": pd.Timestamp("2026-01-05"), "rank": 20, "points": 900},
        ]
    ).to_parquet(rankings_path, index=False)

    schedule_path = tmp_path / "schedule.csv"
    odds_path = tmp_path / "odds.csv"
    pd.DataFrame(
        [
            {
                "match_date": "2026-01-10",
                "tourney_name": "Test Event",
                "surface": "Hard",
                "round": "R16",
                "player_name": "Alpha One",
                "opp_name": "Beta Two",
                "player_rank": 10,
                "opp_rank": 20,
                "best_of": 3,
                "draw_size": 32,
            }
        ]
    ).to_csv(schedule_path, index=False)
    pd.DataFrame(
        [
            {
                "match_date": "2026-01-10",
                "tourney_name": "Test Event",
                "surface": "Hard",
                "round": "R16",
                "player_name": "Alpha One",
                "opp_name": "Beta Two",
                "player_decimal_odds": 1.91,
                "opp_decimal_odds": 1.99,
            }
        ]
    ).to_csv(odds_path, index=False)

    monkeypatch.setenv("TENNIS_SCHEDULE_SOURCE_URI", str(schedule_path))
    monkeypatch.setenv("TENNIS_ODDS_SOURCE_URI", str(odds_path))
    monkeypatch.setenv("TENNIS_PROMOTED_BUNDLE_DIR", str(bundle_dir))
    monkeypatch.setenv("TENNIS_MATCH_DB_PATH", str(matches_path))
    monkeypatch.setenv("TENNIS_RANKINGS_PATH", str(rankings_path))
    monkeypatch.setenv("TENNIS_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("TENNIS_MIN_FEATURE_COVERAGE", "0.0")
    monkeypatch.setenv("TENNIS_UPLOAD_TO_GCS", "false")
    monkeypatch.setattr(
        "src.pipeline.tennis_shadow_ledger.cfg.SHADOW_LEDGER_PATH",
        tmp_path / "ledger" / "shadow_ledger.csv",
    )
    monkeypatch.setattr(
        "src.pipeline.tennis_shadow_ledger.cfg.OPENING_ODDS_DIR",
        tmp_path / "ledger" / "opening_odds",
    )
    monkeypatch.setattr(
        "src.pipeline.tennis_shadow_ledger.cfg.ODDS_SNAPSHOT_DIR",
        tmp_path / "ledger" / "odds_snapshots",
    )
    monkeypatch.setattr(
        "src.pipeline.tennis_shadow_ledger.cfg.CLOSE_ODDS_DIR",
        tmp_path / "ledger" / "close_odds",
    )
    monkeypatch.setattr(
        "src.pipeline.tennis_shadow_ledger.cfg.LEDGER_DIR",
        tmp_path / "ledger",
    )

    monkeypatch.setattr(
        "src.pipeline.tennis_daily_run.load_players",
        lambda verbose=False: pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "name_first": "Alpha",
                    "name_last": "One",
                    "hand": "R",
                    "height": 185,
                    "dob": 19980101,
                },
                {
                    "player_id": 2,
                    "name_first": "Beta",
                    "name_last": "Two",
                    "hand": "R",
                    "height": 188,
                    "dob": 19970101,
                },
            ]
        ),
    )

    manifest = run_daily("2026-01-10")
    assert manifest["status"] == "success"
    assert manifest["row_counts"]["prediction_rows"] == 1
    assert manifest["row_counts"]["shadow_ledger_rows"] == 1
    run_dir = tmp_path / "runs" / "2026" / "01" / "10" / manifest["run_id"]
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "predictions.csv").exists()
    assert (run_dir / "recommendations.csv").exists()
    assert (tmp_path / "ledger" / "shadow_ledger.csv").exists()
    assert (tmp_path / "ledger" / "opening_odds" / "2026-01-10" / "opening.csv").exists()
    assert (tmp_path / "ledger" / "odds_snapshots" / "2026-01-10" / f"{manifest['run_id']}.csv").exists()


def test_close_snapshot_updates_shadow_ledger(tmp_path, monkeypatch):
    feature_cols = get_feature_columns()
    bundle_dir = tmp_path / "promoted" / "current"
    bundle_dir.mkdir(parents=True)

    model = DummyClassifier(strategy="prior")
    X_fit = np.zeros((2, len(feature_cols)))
    y_fit = np.array([0, 1])
    model.fit(X_fit, y_fit)
    scaler = StandardScaler().fit(X_fit)
    medians = {col: 0.0 for col in feature_cols}

    joblib.dump(model, bundle_dir / "model.pkl")
    joblib.dump(scaler, bundle_dir / "scaler.pkl")
    joblib.dump(medians, bundle_dir / "medians.pkl")
    (bundle_dir / "metadata.json").write_text(json.dumps({"val_auc_roc": 0.75}))
    (bundle_dir / "feature_columns.json").write_text(json.dumps(feature_cols))
    (bundle_dir / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "version": "test_bundle",
                "feature_checksum": sha256("\n".join(feature_cols).encode("utf-8")).hexdigest(),
            }
        )
    )

    matches_path = tmp_path / "matches.parquet"
    rankings_path = tmp_path / "rankings.parquet"
    _historical_rows().to_parquet(matches_path, index=False)
    pd.DataFrame(
        [
            {"player_id": 1, "ranking_date": pd.Timestamp("2026-01-05"), "rank": 10, "points": 1000},
            {"player_id": 2, "ranking_date": pd.Timestamp("2026-01-05"), "rank": 20, "points": 900},
        ]
    ).to_parquet(rankings_path, index=False)

    schedule_path = tmp_path / "schedule.csv"
    odds_path = tmp_path / "odds.csv"
    pd.DataFrame(
        [
            {
                "match_date": "2026-01-10",
                "tourney_name": "Test Event",
                "surface": "Hard",
                "round": "R16",
                "player_name": "Alpha One",
                "opp_name": "Beta Two",
                "player_rank": 10,
                "opp_rank": 20,
                "best_of": 3,
                "draw_size": 32,
            }
        ]
    ).to_csv(schedule_path, index=False)
    pd.DataFrame(
        [
            {
                "match_date": "2026-01-10",
                "tourney_name": "Test Event",
                "surface": "Hard",
                "round": "R16",
                "player_name": "Alpha One",
                "opp_name": "Beta Two",
                "player_decimal_odds": 1.91,
                "opp_decimal_odds": 1.99,
            }
        ]
    ).to_csv(odds_path, index=False)

    monkeypatch.setenv("TENNIS_SCHEDULE_SOURCE_URI", str(schedule_path))
    monkeypatch.setenv("TENNIS_ODDS_SOURCE_URI", str(odds_path))
    monkeypatch.setenv("TENNIS_PROMOTED_BUNDLE_DIR", str(bundle_dir))
    monkeypatch.setenv("TENNIS_MATCH_DB_PATH", str(matches_path))
    monkeypatch.setenv("TENNIS_RANKINGS_PATH", str(rankings_path))
    monkeypatch.setenv("TENNIS_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("TENNIS_MIN_FEATURE_COVERAGE", "0.0")
    monkeypatch.setenv("TENNIS_UPLOAD_TO_GCS", "false")
    monkeypatch.setattr(
        "src.pipeline.tennis_shadow_ledger.cfg.SHADOW_LEDGER_PATH",
        tmp_path / "ledger" / "shadow_ledger.csv",
    )
    monkeypatch.setattr(
        "src.pipeline.tennis_shadow_ledger.cfg.OPENING_ODDS_DIR",
        tmp_path / "ledger" / "opening_odds",
    )
    monkeypatch.setattr(
        "src.pipeline.tennis_shadow_ledger.cfg.ODDS_SNAPSHOT_DIR",
        tmp_path / "ledger" / "odds_snapshots",
    )
    monkeypatch.setattr(
        "src.pipeline.tennis_shadow_ledger.cfg.CLOSE_ODDS_DIR",
        tmp_path / "ledger" / "close_odds",
    )
    monkeypatch.setattr(
        "src.pipeline.tennis_shadow_ledger.cfg.LEDGER_DIR",
        tmp_path / "ledger",
    )

    monkeypatch.setattr(
        "src.pipeline.tennis_daily_run.load_players",
        lambda verbose=False: pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "name_first": "Alpha",
                    "name_last": "One",
                    "hand": "R",
                    "height": 185,
                    "dob": 19980101,
                },
                {
                    "player_id": 2,
                    "name_first": "Beta",
                    "name_last": "Two",
                    "hand": "R",
                    "height": 188,
                    "dob": 19970101,
                },
            ]
        ),
    )

    shadow_manifest = run_daily("2026-01-10")
    assert shadow_manifest["status"] == "success"

    pd.DataFrame(
        [
            {
                "match_date": "2026-01-10",
                "tourney_name": "Test Event",
                "surface": "Hard",
                "round": "R16",
                "player_name": "Alpha One",
                "opp_name": "Beta Two",
                "player_decimal_odds": 2.10,
                "opp_decimal_odds": 1.80,
            }
        ]
    ).to_csv(odds_path, index=False)

    close_manifest = run_daily("2026-01-10", mode="close_snapshot")
    assert close_manifest["status"] == "success"
    assert close_manifest["row_counts"]["shadow_ledger_close_rows"] == 1
    assert close_manifest["row_counts"]["shadow_ledger_beat_close_rows"] == 1
    assert (
        tmp_path
        / "ledger"
        / "close_odds"
        / "2026-01-10"
        / f"{close_manifest['run_id']}.csv"
    ).exists()

    ledger = pd.read_csv(tmp_path / "ledger" / "shadow_ledger.csv")
    assert len(ledger) == 1
    assert pd.notna(ledger.loc[0, "closing_pick_market_prob"])
    assert pd.notna(ledger.loc[0, "close_snapshot_utc"])
    assert bool(ledger.loc[0, "beat_close"])
    assert "_close_snapshot_utc_new" not in ledger.columns

    close_manifest_rerun = run_daily("2026-01-10", mode="close_snapshot")
    assert close_manifest_rerun["status"] == "success"
    assert close_manifest_rerun["row_counts"]["shadow_ledger_close_rows"] == 1


def test_shadow_ledger_rerun_replaces_same_match_side(tmp_path, monkeypatch):
    from src.pipeline.tennis_shadow_ledger import record_shadow_ledger

    monkeypatch.setattr(
        "src.pipeline.tennis_shadow_ledger.cfg.SHADOW_LEDGER_PATH",
        tmp_path / "ledger" / "shadow_ledger.csv",
    )
    monkeypatch.setattr(
        "src.pipeline.tennis_shadow_ledger.cfg.LEDGER_DIR",
        tmp_path / "ledger",
    )

    scored = pd.DataFrame(
        [
            {
                "match_id": "match-1",
                "player_name": "Alpha One",
                "opp_name": "Beta Two",
                "side": "player",
                "eligible": True,
                "edge": 0.02,
            }
        ]
    )
    assert record_shadow_ledger("2026-01-10", "run-old", scored) == 1

    scored.loc[0, "edge"] = 0.04
    assert record_shadow_ledger("2026-01-10", "run-new", scored) == 1

    ledger = pd.read_csv(tmp_path / "ledger" / "shadow_ledger.csv")
    assert len(ledger) == 1
    assert ledger.loc[0, "run_id"] == "run-new"
    assert ledger.loc[0, "edge"] == 0.04


def test_daily_run_supports_odds_api_adapter(tmp_path, monkeypatch):
    feature_cols = get_feature_columns()
    bundle_dir = tmp_path / "promoted" / "current"
    bundle_dir.mkdir(parents=True)

    model = DummyClassifier(strategy="prior")
    X_fit = np.zeros((2, len(feature_cols)))
    y_fit = np.array([0, 1])
    model.fit(X_fit, y_fit)
    scaler = StandardScaler().fit(X_fit)
    medians = {col: 0.0 for col in feature_cols}

    joblib.dump(model, bundle_dir / "model.pkl")
    joblib.dump(scaler, bundle_dir / "scaler.pkl")
    joblib.dump(medians, bundle_dir / "medians.pkl")
    (bundle_dir / "metadata.json").write_text(json.dumps({"val_auc_roc": 0.75}))
    (bundle_dir / "feature_columns.json").write_text(json.dumps(feature_cols))
    (bundle_dir / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "version": "test_bundle",
                "feature_checksum": sha256("\n".join(feature_cols).encode("utf-8")).hexdigest(),
            }
        )
    )

    matches_path = tmp_path / "matches.parquet"
    rankings_path = tmp_path / "rankings.parquet"
    _historical_rows().to_parquet(matches_path, index=False)
    pd.DataFrame(
        [
            {"player_id": 1, "ranking_date": pd.Timestamp("2026-01-05"), "rank": 10, "points": 1000},
            {"player_id": 2, "ranking_date": pd.Timestamp("2026-01-05"), "rank": 20, "points": 900},
        ]
    ).to_parquet(rankings_path, index=False)

    schedule_path = tmp_path / "schedule.csv"
    pd.DataFrame(
        [
            {
                "match_date": "2026-01-10",
                "tourney_name": "Madrid Open",
                "surface": "Clay",
                "round": "R16",
                "player_name": "Alpha One",
                "opp_name": "Beta Two",
                "player_rank": 10,
                "opp_rank": 20,
                "best_of": 3,
                "draw_size": 32,
            }
        ]
    ).to_csv(schedule_path, index=False)

    monkeypatch.setenv("TENNIS_SOURCE_ADAPTER", "odds_api_tennis")
    monkeypatch.setenv("TENNIS_SCHEDULE_SOURCE_URI", str(schedule_path))
    monkeypatch.delenv("TENNIS_ODDS_SOURCE_URI", raising=False)
    monkeypatch.setenv("ODDS_API_KEY", "test_key")
    monkeypatch.setenv("TENNIS_PROMOTED_BUNDLE_DIR", str(bundle_dir))
    monkeypatch.setenv("TENNIS_MATCH_DB_PATH", str(matches_path))
    monkeypatch.setenv("TENNIS_RANKINGS_PATH", str(rankings_path))
    monkeypatch.setenv("TENNIS_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("TENNIS_MIN_FEATURE_COVERAGE", "0.0")
    monkeypatch.setenv("TENNIS_UPLOAD_TO_GCS", "false")
    monkeypatch.setattr(
        "src.pipeline.tennis_shadow_ledger.cfg.SHADOW_LEDGER_PATH",
        tmp_path / "ledger" / "shadow_ledger.csv",
    )
    monkeypatch.setattr(
        "src.pipeline.tennis_shadow_ledger.cfg.OPENING_ODDS_DIR",
        tmp_path / "ledger" / "opening_odds",
    )
    monkeypatch.setattr(
        "src.pipeline.tennis_shadow_ledger.cfg.ODDS_SNAPSHOT_DIR",
        tmp_path / "ledger" / "odds_snapshots",
    )
    monkeypatch.setattr(
        "src.pipeline.tennis_shadow_ledger.cfg.CLOSE_ODDS_DIR",
        tmp_path / "ledger" / "close_odds",
    )
    monkeypatch.setattr(
        "src.pipeline.tennis_shadow_ledger.cfg.LEDGER_DIR",
        tmp_path / "ledger",
    )
    monkeypatch.setattr(
        "src.pipeline.tennis_daily_run.load_players",
        lambda verbose=False: pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "name_first": "Alpha",
                    "name_last": "One",
                    "hand": "R",
                    "height": 185,
                    "dob": 19980101,
                },
                {
                    "player_id": 2,
                    "name_first": "Beta",
                    "name_last": "Two",
                    "hand": "R",
                    "height": 188,
                    "dob": 19970101,
                },
            ]
        ),
    )
    monkeypatch.setattr(
        "src.pipeline.tennis_daily_run.fetch_tennis_odds_consensus",
        lambda **kwargs: (
            pd.DataFrame(
                [
                    {
                        "match_id": "oddsapi_fake",
                        "match_date": "2026-01-10T15:00:00Z",
                        "tourney_name": "Madrid Open",
                        "surface": "Clay",
                        "round": "",
                        "player_name": "Alpha One",
                        "opp_name": "Beta Two",
                        "player_decimal_odds": 1.91,
                        "opp_decimal_odds": 1.99,
                        "provider": "odds_api_consensus",
                    }
                ]
            ),
            {
                "raw_payload": b'{"ok": true}',
                "request_headers": [{"sport_key": "tennis_atp_madrid_open", "x_requests_remaining": "499"}],
                "unsupported_tournaments": [],
                "requested_tournaments": ["tennis_atp_madrid_open"],
            },
        ),
    )

    manifest = run_daily("2026-01-10")
    assert manifest["status"] == "success"
    assert manifest["validation_results"]["odds_api"]["requested_tournaments"] == [
        "tennis_atp_madrid_open"
    ]
