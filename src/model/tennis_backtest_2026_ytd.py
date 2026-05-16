"""
Approximate 2026 YTD tennis backtest using the current production model.

Why approximate:
  - Sackmann result files are not yet available for 2025/2026 in this repo's source.
  - We extend match history with tennis-data.co.uk rows for 2025 and 2026.
  - Those rows have outcomes, ranks, and odds, but not the full point-level stat set.
  - Serve/return rolling features for 2025/2026 therefore become sparse and are
    filled using the production median artifacts saved at train time.

This is still a useful forward-style test for the model entering 2026, but it is
lower confidence than the historical 2022-2024 backtests built on full Sackmann rows.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config_tennis as cfg
from src.data.tennis_odds import devig_multiplicative
from src.data.tennis_sackmann import load_matches, load_players
from src.data.tennis_tml import load_tml_player_rows
from src.features.tennis_engineer import build_features, get_feature_columns
from src.model.tennis_oos_report import (
    _score_picks,
    _simulate_flat_bet,
    _simulate_kelly,
)

OUT_DIR = cfg.OOS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

_TD_YEARS = (2025, 2026)


def _canon(text: str) -> str:
    return re.sub(r"[^a-z]", "", str(text).lower())


def _series_to_level(series: str, tournament: str) -> str:
    s = str(series).strip().lower()
    t = str(tournament).strip().lower()
    if "grand slam" in s or t in {"wimbledon", "us open", "australian open", "french open", "roland garros"}:
        return "G"
    if "masters 1000" in s:
        return "M"
    if "atp250" in s or "atp500" in s or "atp finals" in s or "united cup" in t:
        return "A"
    return "A"


def _round_text_to_code(round_text: str, level: str, first_round_matches: int) -> str:
    text = str(round_text).strip().lower()
    big_draw = first_round_matches >= 28 or level == "G"
    if text == "1st round":
        return "R128" if level == "G" else ("R64" if big_draw else "R32")
    if text == "2nd round":
        return "R64" if level == "G" else ("R32" if big_draw else "R16")
    if text == "3rd round":
        return "R32" if level == "G" else ("R16" if big_draw else "QF")
    if text == "4th round":
        return "R16" if level == "G" else "QF"
    if text == "quarterfinals":
        return "QF"
    if text == "semifinals":
        return "SF"
    if text == "the final":
        return "F"
    if text == "round robin":
        return "RR"
    return "R32"


def _score_string(row: pd.Series) -> str:
    sets = []
    for i in range(1, 6):
        w = row.get(f"W{i}")
        l = row.get(f"L{i}")
        if pd.notna(w) and pd.notna(l):
            sets.append(f"{int(w)}-{int(l)}")
    return " ".join(sets)


@dataclass
class PlayerRecord:
    player_id: int
    player_name: str
    player_hand: str | float
    player_ht: float
    player_age: float


def _build_player_lookup(matches_hist: pd.DataFrame, players_master: pd.DataFrame) -> tuple[dict[str, list[dict]], dict[int, int]]:
    hist_counts = matches_hist["player_id"].value_counts().to_dict()
    master = players_master.copy()
    master["surname_key"] = master["name_last"].map(_canon)
    master["first_key"] = master["name_first"].map(_canon)
    master["initials_key"] = master["name_first"].map(
        lambda s: "".join(p[0] for p in re.split(r"[^A-Za-z]+", str(s).lower()) if p)
    )
    master["player_name"] = master["name_first"].fillna("").astype(str).str.strip() + " " + master["name_last"].fillna("").astype(str).str.strip()

    lookup: dict[str, list[dict]] = {}
    for rec in master.to_dict("records"):
        lookup.setdefault(rec["surname_key"], []).append(rec)
    return lookup, hist_counts


def _resolve_player(
    td_name: str,
    match_date: pd.Timestamp,
    lookup: dict[str, list[dict]],
    hist_counts: dict[int, int],
    synthetic_ids: dict[str, int],
) -> PlayerRecord:
    cleaned = str(td_name).strip().replace(".", "")
    parts = cleaned.split()
    surname_key = _canon(" ".join(parts[:-1])) if len(parts) >= 2 else _canon(cleaned)
    token = _canon(parts[-1]) if len(parts) >= 2 else ""
    candidates = lookup.get(surname_key, [])

    if candidates:
        filtered = [
            rec for rec in candidates
            if str(rec["initials_key"]).startswith(token) or str(rec["first_key"]).startswith(token)
        ]
        if not filtered:
            filtered = candidates
        filtered = sorted(filtered, key=lambda rec: hist_counts.get(int(rec["player_id"]), 0), reverse=True)
        chosen = filtered[0]
        dob = pd.to_datetime(str(int(chosen["dob"])) if pd.notna(chosen["dob"]) else "", format="%Y%m%d", errors="coerce")
        age = ((match_date - dob).days / 365.25) if pd.notna(dob) else np.nan
        return PlayerRecord(
            player_id=int(chosen["player_id"]),
            player_name=chosen["player_name"].strip(),
            player_hand=chosen.get("hand", np.nan),
            player_ht=float(chosen["height"]) if pd.notna(chosen.get("height")) else np.nan,
            player_age=age,
        )

    key = _canon(cleaned)
    if key not in synthetic_ids:
        synthetic_ids[key] = 900_000_000 + len(synthetic_ids) + 1
    display_name = " ".join(parts[-1:] + parts[:-1]) if len(parts) >= 2 else cleaned
    return PlayerRecord(
        player_id=synthetic_ids[key],
        player_name=display_name,
        player_hand=np.nan,
        player_ht=np.nan,
        player_age=np.nan,
    )


def _load_tennis_data_year(year: int) -> pd.DataFrame:
    url = f"http://www.tennis-data.co.uk/{year}/{year}.xlsx"
    df = pd.read_excel(url)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()].copy()
    df = df[df["Comment"].fillna("Completed").astype(str).str.lower().ne("cancelled")].copy()
    return df


def _build_recent_extension(matches_hist: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    players_master = load_players(verbose=False)
    lookup, hist_counts = _build_player_lookup(matches_hist, players_master)
    synthetic_ids: dict[str, int] = {}
    event_frames = []

    for year in _TD_YEARS:
        td = _load_tennis_data_year(year)
        td["week_start"] = td["Date"] - pd.to_timedelta(td["Date"].dt.weekday, unit="D")
        first_round_counts = (
            td[td["Round"].astype(str).str.lower() == "1st round"]
            .groupby(["Tournament", "week_start"])
            .size()
            .to_dict()
        )
        rows = []
        for row in td.to_dict("records"):
            match_date = pd.Timestamp(row["Date"])
            level = _series_to_level(row.get("Series", ""), row.get("Tournament", ""))
            fr_count = first_round_counts.get((row["Tournament"], row["week_start"]), 16)
            round_code = _round_text_to_code(row.get("Round", ""), level, fr_count)
            draw_size = 128 if level == "G" else (64 if fr_count >= 28 else 32)
            winner = _resolve_player(row["Winner"], match_date, lookup, hist_counts, synthetic_ids)
            loser = _resolve_player(row["Loser"], match_date, lookup, hist_counts, synthetic_ids)
            score = _score_string(pd.Series(row))
            comment = str(row.get("Comment", "Completed"))
            is_wo = "walkover" in comment.lower()
            is_ret = "ret" in comment.lower() or "retired" in comment.lower()
            base = {
                "match_id": f"td_{match_date.date()}_{_canon(row['Tournament'])}_{winner.player_id}_{loser.player_id}",
                "tourney_date": match_date,
                "tourney_name": row["Tournament"],
                "surface": row.get("Surface", ""),
                "tourney_level": level,
                "draw_size": draw_size,
                "round": round_code,
                "best_of": int(row.get("Best of", 3)) if pd.notna(row.get("Best of")) else 3,
                "score": score,
                "minutes": np.nan,
                "is_walkover": is_wo,
                "is_retirement": is_ret,
                "data_source": "tennis_data_recent",
                "sim_date": match_date,
                "td_winner_odds": row.get("PSW", row.get("B365W", np.nan)),
                "td_loser_odds": row.get("PSL", row.get("B365L", np.nan)),
                "td_b365_winner_odds": row.get("B365W", np.nan),
                "td_b365_loser_odds": row.get("B365L", np.nan),
            }
            winner_row = {
                **base,
                "player_id": winner.player_id,
                "player_name": winner.player_name,
                "player_hand": winner.player_hand,
                "player_ht": winner.player_ht,
                "player_age": winner.player_age,
                "player_rank": pd.to_numeric(row.get("WRank"), errors="coerce"),
                "player_rank_points": pd.to_numeric(row.get("WPts"), errors="coerce"),
                "opp_id": loser.player_id,
                "opp_name": loser.player_name,
                "opp_hand": loser.player_hand,
                "opp_ht": loser.player_ht,
                "opp_age": loser.player_age,
                "opp_rank": pd.to_numeric(row.get("LRank"), errors="coerce"),
                "opp_rank_points": pd.to_numeric(row.get("LPts"), errors="coerce"),
                "won": 1,
            }
            loser_row = {
                **base,
                "player_id": loser.player_id,
                "player_name": loser.player_name,
                "player_hand": loser.player_hand,
                "player_ht": loser.player_ht,
                "player_age": loser.player_age,
                "player_rank": pd.to_numeric(row.get("LRank"), errors="coerce"),
                "player_rank_points": pd.to_numeric(row.get("LPts"), errors="coerce"),
                "opp_id": winner.player_id,
                "opp_name": winner.player_name,
                "opp_hand": winner.player_hand,
                "opp_ht": winner.player_ht,
                "opp_age": winner.player_age,
                "opp_rank": pd.to_numeric(row.get("WRank"), errors="coerce"),
                "opp_rank_points": pd.to_numeric(row.get("WPts"), errors="coerce"),
                "won": 0,
            }
            for stat_col in [
                "ace", "df", "svpt", "first_in", "first_won", "second_won", "sv_gms",
                "bp_saved", "bp_faced", "opp_ace", "opp_df", "opp_svpt", "opp_first_in",
                "opp_first_won", "opp_second_won", "opp_sv_gms", "opp_bp_saved", "opp_bp_faced",
            ]:
                winner_row[stat_col] = np.nan
                loser_row[stat_col] = np.nan
            rows.extend([winner_row, loser_row])
        event_frames.append(pd.DataFrame(rows))

    combined = pd.concat(event_frames, ignore_index=True)
    return combined, players_master


def _deduplicate_to_one_row_per_match(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_rank_p"] = df["player_rank"].fillna(9999)
    df["_rank_o"] = df["opp_rank"].fillna(9999)
    df["_pid"] = pd.to_numeric(df["player_id"], errors="coerce").fillna(9999999)
    df["_oid"] = pd.to_numeric(df["opp_id"], errors="coerce").fillna(9999999)
    df["_keep"] = (df["_rank_p"] < df["_rank_o"]) | (
        (df["_rank_p"] == df["_rank_o"]) & (df["_pid"] <= df["_oid"])
    )
    df = df.sort_values(["match_id", "_keep"], ascending=[True, False])
    kept = df[df["_keep"]].drop_duplicates(subset=["match_id"]).copy()
    return kept.drop(columns=["_rank_p", "_rank_o", "_pid", "_oid", "_keep"])


def build_extended_matches(output_path: Path | None = None) -> Path:
    """Build historical matches extended with recent 2025/2026 completed rows.

    Prefer TennisMyLife rows because they include the Sackmann-style serve/return
    stat columns needed by rolling features. Fall back to the older tennis-data
    approximation only if the recent feed is unavailable.
    """
    print("Loading historical player-perspective match DB (through 2024)...", flush=True)
    hist = load_matches()
    hist = hist[hist["tourney_date"].dt.year <= 2024].copy()

    print(f"  {len(hist)} historical rows. Extending with recent 2025/2026 match history...", flush=True)
    recent_rows = load_tml_player_rows([2025, 2026], include_challenger=True, verbose=True)
    if recent_rows.empty:
        print("  TennisMyLife recent rows unavailable. Falling back to tennis-data approximation...", flush=True)
        recent_rows, _ = _build_recent_extension(hist)
    else:
        recent_rows = recent_rows[recent_rows["tourney_date"].dt.year.isin([2025, 2026])].copy()
    print(f"  {len(recent_rows)} recent rows added.", flush=True)

    extended = pd.concat([hist, recent_rows], ignore_index=True, sort=False)
    extended = extended.drop_duplicates(subset=["match_id", "player_id", "opp_id"], keep="last")
    # Coerce odds columns to float; tennis-data uses '-' for missing values
    for col in ["td_winner_odds", "td_loser_odds", "td_b365_winner_odds", "td_b365_loser_odds"]:
        if col in extended.columns:
            extended[col] = pd.to_numeric(extended[col], errors="coerce")

    if output_path is None:
        ledger_dir = Path(cfg.LEDGER_DIR)
        ledger_dir.mkdir(parents=True, exist_ok=True)
        output_path = ledger_dir / "extended_matches.parquet"

    extended.to_parquet(output_path, index=False)
    print(f"  Extended match DB saved → {output_path} ({len(extended)} rows)", flush=True)
    return output_path


def build_2026_ytd_backtest(output_tag: str | None = None) -> dict:
    print("Loading historical player-perspective match DB...")
    hist = load_matches()
    hist = hist[hist["tourney_date"].dt.year <= 2024].copy()
    print(f"  historical rows through 2024: {len(hist)}")

    print("Building 2025-2026 extension from tennis-data...")
    recent_rows, _ = _build_recent_extension(hist)
    print(f"  recent rows: {len(recent_rows)}")

    combined = pd.concat([hist, recent_rows], ignore_index=True, sort=False)
    print("Building features over combined history...")
    features = build_features(combined, verbose=True)

    print("Loading production model artifacts...")
    model = joblib.load(cfg.MODEL_PATH)
    scaler = joblib.load(cfg.SCALER_PATH)
    medians = joblib.load(cfg.MEDIANS_PATH)

    feat_cols = get_feature_columns()
    pred_df = features[features["tourney_date"].dt.year == 2026].copy()
    pred_df = _deduplicate_to_one_row_per_match(pred_df)
    fill_values = pd.Series(medians)
    pred_df[feat_cols] = pred_df[feat_cols].fillna(fill_values)
    X = scaler.transform(pred_df[feat_cols].values)
    pred_df["player_win_prob_raw"] = model.predict_proba(X)[:, 1]
    pred_df["player_win_prob"] = pred_df["player_win_prob_raw"]

    pred_df["winner_name_odds"] = np.where(pred_df["won"] == 1, pred_df["player_name"], pred_df["opp_name"])
    pred_df["loser_name_odds"] = np.where(pred_df["won"] == 1, pred_df["opp_name"], pred_df["player_name"])
    pred_df["player_model_prob"] = pred_df["player_win_prob"]
    pred_df["opp_model_prob"] = 1.0 - pred_df["player_win_prob"]
    pred_df["player_model_prob_raw"] = pred_df["player_win_prob_raw"]
    pred_df["opp_model_prob_raw"] = 1.0 - pred_df["player_win_prob_raw"]

    pred_df["pinnacle_winner_odds"] = pd.to_numeric(pred_df["td_winner_odds"], errors="coerce")
    pred_df["pinnacle_loser_odds"] = pd.to_numeric(pred_df["td_loser_odds"], errors="coerce")
    pred_df["b365_winner_odds"] = pd.to_numeric(pred_df["td_b365_winner_odds"], errors="coerce")
    pred_df["b365_loser_odds"] = pd.to_numeric(pred_df["td_b365_loser_odds"], errors="coerce")

    pred_df["winner_decimal_odds"] = pred_df["pinnacle_winner_odds"].where(
        pred_df["pinnacle_winner_odds"].notna() & (pred_df["pinnacle_winner_odds"] > 1.0),
        pred_df["b365_winner_odds"],
    )
    pred_df["loser_decimal_odds"] = pred_df["pinnacle_loser_odds"].where(
        pred_df["pinnacle_loser_odds"].notna() & (pred_df["pinnacle_loser_odds"] > 1.0),
        pred_df["b365_loser_odds"],
    )

    novig = pred_df.apply(
        lambda r: pd.Series(devig_multiplicative(r["winner_decimal_odds"], r["loser_decimal_odds"])),
        axis=1,
    )
    pred_df[["novig_winner_prob", "novig_loser_prob"]] = novig.values

    player_is_winner = pred_df["won"] == 1
    pred_df["player_market_prob"] = np.where(player_is_winner, pred_df["novig_winner_prob"], pred_df["novig_loser_prob"])
    pred_df["opp_market_prob"] = np.where(player_is_winner, pred_df["novig_loser_prob"], pred_df["novig_winner_prob"])
    pred_df["player_decimal_odds"] = np.where(player_is_winner, pred_df["winner_decimal_odds"], pred_df["loser_decimal_odds"])
    pred_df["opp_decimal_odds"] = np.where(player_is_winner, pred_df["loser_decimal_odds"], pred_df["winner_decimal_odds"])
    pred_df["winner_rank"] = np.where(player_is_winner, pred_df["player_rank"], pred_df["opp_rank"])
    pred_df["loser_rank"] = np.where(player_is_winner, pred_df["opp_rank"], pred_df["player_rank"])
    pred_df["match_date_odds"] = pred_df["sim_date"]

    scored = _score_picks(pred_df, edge_threshold=cfg.EDGE_THRESHOLD)
    picks = scored[scored["eligible"]].copy()
    flat_stats = _simulate_flat_bet(picks)
    kelly_df = _simulate_kelly(picks, cfg.STARTING_BANKROLL, cfg.KELLY_FRACTION)

    suffix = f"_{output_tag}" if output_tag else ""
    scored_path = OUT_DIR / f"scored_oos_2026_ytd{suffix}.csv"
    kelly_path = OUT_DIR / f"kelly_oos_2026_ytd{suffix}.csv"
    report_path = OUT_DIR / f"oos_report_2026_ytd{suffix}.json"
    scored.to_csv(scored_path, index=False)
    kelly_df.to_csv(kelly_path, index=False)

    total_bets = len(kelly_df)
    wins = int(kelly_df["won"].sum()) if total_bets else 0
    total_profit = float(kelly_df["profit"].sum()) if total_bets else 0.0
    total_staked = float(kelly_df["stake"].sum()) if total_bets else 1.0

    summary = {
        "oos_years": [2026],
        "backtest_note": "Approximate 2026 YTD backtest using tennis-data match history for 2025-2026.",
        "model_artifact": str(cfg.MODEL_PATH),
        "feature_medians_artifact": str(cfg.MEDIANS_PATH),
        "edge_threshold": cfg.EDGE_THRESHOLD,
        "kelly_fraction": cfg.KELLY_FRACTION,
        "starting_bankroll": cfg.STARTING_BANKROLL,
        "total_matches_with_odds": int(len(scored)),
        "governance_blocked": int(scored["governance_block"].notna().sum()),
        "flat_bet": flat_stats,
        "total_bets": total_bets,
        "win_rate": round(wins / total_bets, 4) if total_bets else None,
        "total_profit": round(total_profit, 2),
        "roi": round(total_profit / total_staked, 4) if total_staked > 0 else None,
        "final_bankroll": round(float(kelly_df["bankroll"].iloc[-1]), 2) if total_bets else cfg.STARTING_BANKROLL,
        "scored_path": str(scored_path),
        "kelly_path": str(kelly_path),
    }
    report_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    summary = build_2026_ytd_backtest()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
