"""Edge diagnostics and simple governance variant comparisons for tennis OOS files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import config_tennis as cfg
from src.model.tennis_oos_report import _simulate_kelly

OOS_DIR = cfg.OOS_DIR


def _scored_path(year: int) -> Path:
    return OOS_DIR / f"scored_oos_{year}_holdout_{year}.csv"


def load_scored_holdouts(years: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in years:
        path = _scored_path(year)
        if not path.exists():
            raise FileNotFoundError(f"Scored holdout file not found: {path}")
        df = pd.read_csv(path)
        df["holdout_year"] = year
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["decimal_odds"] = pd.to_numeric(out["decimal_odds"], errors="coerce")
    out["sim_date"] = pd.to_datetime(out["sim_date"], errors="coerce")
    out["fav_dog"] = out["decimal_odds"].apply(lambda x: "fav" if pd.notna(x) and x < 2.0 else "dog")
    out["odds_band"] = pd.cut(
        out["decimal_odds"],
        bins=[1.0, 1.5, 2.0, 3.0, 100.0],
        labels=["<=1.5", "1.5-2.0", "2.0-3.0", "3.0+"],
        include_lowest=True,
    )
    return out


def _eligible(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["eligible"]].copy()
    out = out[out["decimal_odds"].notna()].copy()
    return out


def _apply_variant(df: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, float]:
    picks = _eligible(df)
    kelly_fraction = cfg.KELLY_FRACTION

    if variant == "current":
        pass
    elif variant == "current_kelly_0.20":
        kelly_fraction = 0.20
    elif variant == "no_masters":
        picks = picks[picks["tourney_level"] != "M"].copy()
    elif variant == "no_masters_kelly_0.20":
        picks = picks[picks["tourney_level"] != "M"].copy()
        kelly_fraction = 0.20
    elif variant == "no_masters_dogs":
        picks = picks[~((picks["tourney_level"] == "M") & (picks["decimal_odds"] >= 2.0))].copy()
    elif variant == "no_masters_dogs_kelly_0.20":
        picks = picks[~((picks["tourney_level"] == "M") & (picks["decimal_odds"] >= 2.0))].copy()
        kelly_fraction = 0.20
    else:
        raise ValueError(f"Unknown variant: {variant}")

    picks = picks.copy()
    picks["eligible"] = True
    return picks, kelly_fraction


def _flat_profit(sub: pd.DataFrame) -> float:
    return float((((sub["decimal_odds"] - 1.0) * sub["won_bet"]) - (1.0 - sub["won_bet"])).sum())


def _flat_summary(picks: pd.DataFrame) -> dict:
    if picks.empty:
        return {"bets": 0, "flat_roi": None, "win_rate": None, "units_profit": 0.0}
    profit = _flat_profit(picks)
    return {
        "bets": int(len(picks)),
        "flat_roi": round(profit / len(picks), 4),
        "win_rate": round(float(picks["won_bet"].mean()), 4),
        "units_profit": round(profit, 2),
    }


def _kelly_summary(picks: pd.DataFrame, kelly_fraction: float) -> tuple[dict, pd.DataFrame]:
    if picks.empty:
        return {
            "bets": 0,
            "kelly_roi": None,
            "total_profit": 0.0,
            "total_staked": 0.0,
            "final_bankroll": cfg.STARTING_BANKROLL,
            "avg_stake": None,
        }, pd.DataFrame()

    kelly_df = _simulate_kelly(
        picks,
        starting_bankroll=cfg.STARTING_BANKROLL,
        kelly_fraction=kelly_fraction,
    )
    total_staked = float(kelly_df["stake"].sum()) if not kelly_df.empty else 0.0
    total_profit = float(kelly_df["profit"].sum()) if not kelly_df.empty else 0.0
    roi = total_profit / total_staked if total_staked > 0 else None
    summary = {
        "bets": int(len(kelly_df)),
        "kelly_roi": round(roi, 4) if roi is not None else None,
        "total_profit": round(total_profit, 2),
        "total_staked": round(total_staked, 2),
        "final_bankroll": round(float(kelly_df["bankroll"].iloc[-1]), 2) if not kelly_df.empty else cfg.STARTING_BANKROLL,
        "avg_stake": round(float(kelly_df["stake"].mean()), 2) if not kelly_df.empty else None,
    }
    return summary, kelly_df


def _group_panel(picks: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict] = []
    for key, sub in picks.groupby(group_col, dropna=False, observed=False):
        if len(sub) == 0:
            continue
        profit = _flat_profit(sub)
        rows.append(
            {
                "split": group_col,
                "group": str(key),
                "bets": int(len(sub)),
                "win_rate": round(float(sub["won_bet"].mean()), 4),
                "flat_roi": round(profit / len(sub), 4),
                "avg_edge": round(float(sub["edge"].mean()), 4),
                "avg_odds": round(float(sub["decimal_odds"].mean()), 4),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "bets"], ascending=[True, False])


def build_edge_diagnostics(
    years: list[int],
    variants: list[str] | None = None,
) -> dict:
    variants = variants or [
        "current",
        "current_kelly_0.20",
        "no_masters",
        "no_masters_kelly_0.20",
        "no_masters_dogs",
        "no_masters_dogs_kelly_0.20",
    ]
    scored = load_scored_holdouts(years)

    variant_rows: list[dict] = []
    panel_frames: list[pd.DataFrame] = []

    for variant in variants:
        picks, kelly_fraction = _apply_variant(scored, variant)
        flat = _flat_summary(picks)
        kelly, _ = _kelly_summary(picks, kelly_fraction)
        blocked_vs_current = int(_eligible(scored).shape[0] - picks.shape[0])
        variant_rows.append(
            {
                "variant": variant,
                "years": years,
                "bets": flat["bets"],
                "blocked_vs_current": blocked_vs_current,
                "flat_roi": flat["flat_roi"],
                "flat_units_profit": flat["units_profit"],
                "flat_win_rate": flat["win_rate"],
                "kelly_fraction": kelly_fraction,
                "kelly_roi": kelly["kelly_roi"],
                "kelly_total_profit": kelly["total_profit"],
                "kelly_total_staked": kelly["total_staked"],
                "final_bankroll": kelly["final_bankroll"],
                "avg_stake": kelly["avg_stake"],
            }
        )

        for split in ["holdout_year", "edge_bucket", "tourney_level", "fav_dog", "odds_band"]:
            panel = _group_panel(picks, split)
            if panel.empty:
                continue
            panel.insert(0, "variant", variant)
            panel_frames.append(panel)

    variant_df = pd.DataFrame(variant_rows).sort_values("flat_roi", ascending=False)
    panel_df = pd.concat(panel_frames, ignore_index=True) if panel_frames else pd.DataFrame()

    tag = "_".join(str(y) for y in years)
    summary_path = OOS_DIR / f"edge_diagnostics_summary_{tag}.json"
    variants_path = OOS_DIR / f"edge_diagnostics_variants_{tag}.csv"
    panel_path = OOS_DIR / f"edge_diagnostics_panels_{tag}.csv"

    summary = {
        "years": years,
        "variants": variant_rows,
        "summary_csv": str(variants_path),
        "panel_csv": str(panel_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    variant_df.to_csv(variants_path, index=False)
    if not panel_df.empty:
        panel_df.to_csv(panel_path, index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tennis edge diagnostics on saved scored OOS files.")
    parser.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024])
    args = parser.parse_args()
    summary = build_edge_diagnostics(args.years)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
