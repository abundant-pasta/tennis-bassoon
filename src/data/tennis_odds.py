"""
Parse tennis-data.co.uk odds files and join to the match DB.

tennis-data.co.uk distributes one Excel/CSV file per year per tour.
ATP URLs follow the pattern:
  http://www.tennis-data.co.uk/{year}/{year}.xlsx   (2011+)
  http://www.tennis-data.co.uk/{year}/{year}.xls    (older)

Relevant columns in the raw file:
  Winner, Loser, Date, Tournament, Surface, Round, Best of
  B365W, B365L  — Bet365 decimal odds (winner / loser)
  PSW,  PSL     — Pinnacle decimal odds
  MaxW, MaxL    — Best available

Output: odds_df joined onto match records by (tourney_date, winner_name, loser_name)
using fuzzy name matching as a fallback.
"""

from __future__ import annotations

import io
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import config_tennis as cfg

RAW_DIR = cfg.RAW_DIR
RAW_DIR.mkdir(parents=True, exist_ok=True)

_TD_BASE = "http://www.tennis-data.co.uk"
_XLSX_YEARS = list(range(2011, 2027))
_XLS_YEARS = list(range(2000, 2011))


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _fetch_bytes(url: str) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read()
    except Exception:
        return None


def _fetch_year(year: int) -> pd.DataFrame | None:
    for ext in ("xlsx", "xls"):
        local = RAW_DIR / f"tennis_data_{year}.{ext}"
        if local.exists():
            try:
                return pd.read_excel(local, engine="openpyxl" if ext == "xlsx" else "xlrd")
            except Exception:
                pass
        url = f"{_TD_BASE}/{year}/{year}.{ext}"
        data = _fetch_bytes(url)
        if data:
            local.write_bytes(data)
            try:
                return pd.read_excel(io.BytesIO(data), engine="openpyxl" if ext == "xlsx" else "xlrd")
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

_COL_RENAME = {
    "Winner": "winner_name_odds",
    "Loser": "loser_name_odds",
    "Date": "match_date_odds",
    "Tournament": "tourney_name_odds",
    "Surface": "surface_odds",
    "Round": "round_odds",
    "Best of": "best_of_odds",
    # Pinnacle
    "PSW": "pinnacle_winner_odds",
    "PSL": "pinnacle_loser_odds",
    # Bet365
    "B365W": "b365_winner_odds",
    "B365L": "b365_loser_odds",
    # Best available
    "MaxW": "max_winner_odds",
    "MaxL": "max_loser_odds",
}


def _parse_year_df(raw: pd.DataFrame, year: int) -> pd.DataFrame:
    raw = raw.copy()
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw.rename(columns={k: v for k, v in _COL_RENAME.items() if k in raw.columns})

    if "match_date_odds" not in raw.columns:
        return pd.DataFrame()

    raw["match_date_odds"] = pd.to_datetime(raw["match_date_odds"], errors="coerce")
    raw = raw.dropna(subset=["match_date_odds", "winner_name_odds", "loser_name_odds"])

    for col in ["pinnacle_winner_odds", "pinnacle_loser_odds",
                "b365_winner_odds", "b365_loser_odds",
                "max_winner_odds", "max_loser_odds"]:
        if col not in raw.columns:
            raw[col] = np.nan
        else:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw["year"] = year
    return raw[[
        "match_date_odds", "winner_name_odds", "loser_name_odds",
        "tourney_name_odds", "surface_odds", "round_odds", "best_of_odds",
        "pinnacle_winner_odds", "pinnacle_loser_odds",
        "b365_winner_odds", "b365_loser_odds",
        "max_winner_odds", "max_loser_odds",
        "year",
    ]]


def load_all_odds(verbose: bool = True) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in _XLSX_YEARS + _XLS_YEARS:
        raw = _fetch_year(year)
        if raw is None:
            if verbose:
                print(f"  odds {year}: not found")
            continue
        parsed = _parse_year_df(raw, year)
        if not parsed.empty:
            frames.append(parsed)
            if verbose:
                print(f"  odds {year}: {len(parsed)} rows")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# De-vig (multiplicative method)
# ---------------------------------------------------------------------------

def devig_multiplicative(winner_decimal: float, loser_decimal: float) -> tuple[float, float]:
    """Convert decimal odds to no-vig implied probabilities via multiplicative method."""
    if winner_decimal <= 1.0 or loser_decimal <= 1.0:
        return np.nan, np.nan
    p_w = 1.0 / winner_decimal
    p_l = 1.0 / loser_decimal
    total = p_w + p_l
    return p_w / total, p_l / total


def add_novig_probs(odds_df: pd.DataFrame) -> pd.DataFrame:
    """Add no-vig winner/loser probabilities from Pinnacle odds (preferred) or Bet365."""
    df = odds_df.copy()

    pin_mask = df["pinnacle_winner_odds"].notna() & df["pinnacle_loser_odds"].notna()
    df[["novig_winner_prob", "novig_loser_prob"]] = np.nan
    if pin_mask.any():
        rows = df.loc[pin_mask, ["pinnacle_winner_odds", "pinnacle_loser_odds"]].apply(
            lambda r: pd.Series(devig_multiplicative(r["pinnacle_winner_odds"], r["pinnacle_loser_odds"])),
            axis=1,
        )
        df.loc[pin_mask, ["novig_winner_prob", "novig_loser_prob"]] = rows.values

    b365_mask = (~pin_mask) & df["b365_winner_odds"].notna() & df["b365_loser_odds"].notna()
    if b365_mask.any():
        rows = df.loc[b365_mask, ["b365_winner_odds", "b365_loser_odds"]].apply(
            lambda r: pd.Series(devig_multiplicative(r["b365_winner_odds"], r["b365_loser_odds"])),
            axis=1,
        )
        df.loc[b365_mask, ["novig_winner_prob", "novig_loser_prob"]] = rows.values

    return df


# ---------------------------------------------------------------------------
# Fuzzy name matching
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    """Extract the last name for fuzzy matching between Sackmann and tennis-data formats.

    Sackmann uses "First Last"; tennis-data uses "Last F." or "Last F.M.".
    Both formats place the primary surname first or last — we extract it reliably:
      - If the first token is longer than the last token(s), it's "Last Initial" → take token[0]
      - Otherwise it's "First Last" → take the last token
    Falls back to full alpha-only name if single word.
    """
    name = str(name).strip()
    alpha_only = re.sub(r"[^a-zA-Z\s]", " ", name).strip()
    parts = alpha_only.split()
    if not parts:
        return name.lower()
    if len(parts) == 1:
        return parts[0].lower()

    # tennis-data format: "Kwon SW" or "Zverev A" — first token is the surname (longer)
    # Sackmann format: "SoonWoo Kwon" or "Alexander Zverev" — last token is the surname (longer)
    first_len = len(parts[0])
    last_len = len(parts[-1])
    # If the last token is 1-2 chars it's an initial → "Last Initial" format (tennis-data)
    if last_len <= 2:
        return parts[0].lower()
    # Otherwise "First Last" format (Sackmann)
    return parts[-1].lower()


def join_odds_to_matches(matches: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Join odds to match rows on (year, month, winner_last_name, loser_last_name).

    Sackmann's tourney_date is the tournament START date, not the individual
    match date, so exact-date joins fail. Matching on year+month + normalized
    player names is robust since two players rarely meet twice in one month.
    """
    if odds.empty:
        for col in ["pinnacle_winner_odds", "pinnacle_loser_odds",
                    "b365_winner_odds", "b365_loser_odds",
                    "novig_winner_prob", "novig_loser_prob"]:
            matches[col] = np.nan
        return matches

    odds = add_novig_probs(odds)
    odds["_wn"] = odds["winner_name_odds"].apply(_normalize_name)
    odds["_ln"] = odds["loser_name_odds"].apply(_normalize_name)
    odds["_ym"] = odds["match_date_odds"].dt.to_period("M").astype(str)

    matches = matches.copy()
    matches["_wn"] = matches["winner_name"].apply(_normalize_name)
    matches["_ln"] = matches["loser_name"].apply(_normalize_name)
    # Use tournament start date to derive year-month (match is in same or next month)
    matches["_tourney_dt"] = pd.to_datetime(matches["tourney_date"])
    matches["_ym"] = matches["_tourney_dt"].dt.to_period("M").astype(str)

    keep_odds = ["_ym", "_wn", "_ln",
                 "pinnacle_winner_odds", "pinnacle_loser_odds",
                 "b365_winner_odds", "b365_loser_odds",
                 "novig_winner_prob", "novig_loser_prob",
                 "match_date_odds"]
    # Deduplicate odds in case same matchup appears twice (same month, different rounds — unlikely)
    odds_dedup = odds[keep_odds].drop_duplicates(subset=["_ym", "_wn", "_ln"], keep="last")

    merged = matches.merge(odds_dedup, on=["_ym", "_wn", "_ln"], how="left")
    merged = merged.reset_index(drop=True)

    # Second pass: try next month for matches in tournaments spanning a month boundary
    fill_cols = ["pinnacle_winner_odds", "pinnacle_loser_odds",
                 "b365_winner_odds", "b365_loser_odds",
                 "novig_winner_prob", "novig_loser_prob", "match_date_odds"]
    unmatched_idx = merged.index[merged["novig_winner_prob"].isna()]
    if len(unmatched_idx) > 0:
        unmatched = merged.loc[unmatched_idx, ["_tourney_dt", "_wn", "_ln"]].copy()
        unmatched["_orig_idx"] = unmatched.index  # preserve row identity through merge
        unmatched["_ym"] = (unmatched["_tourney_dt"] + pd.DateOffset(months=1)).dt.to_period("M").astype(str)
        extra = unmatched.merge(odds_dedup, on=["_ym", "_wn", "_ln"], how="inner")
        if not extra.empty:
            extra = extra.set_index("_orig_idx")
            for col in fill_cols:
                if col in extra.columns:
                    merged.loc[extra.index, col] = extra[col]

    merged.drop(columns=["_wn", "_ln", "_ym", "_tourney_dt"], inplace=True)
    return merged


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_odds(odds: pd.DataFrame) -> None:
    out = cfg.FEATURES_DIR / "odds.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    odds.to_parquet(out, index=False)
    print(f"Saved {len(odds)} odds rows → {out}")


def load_odds() -> pd.DataFrame:
    path = cfg.FEATURES_DIR / "odds.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Odds parquet not found at {path}.")
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    print("Downloading tennis-data.co.uk odds files...")
    odds = load_all_odds(verbose=True)
    if odds.empty:
        print("No odds data retrieved.")
        return
    print(f"\nTotal odds rows: {len(odds)}")
    save_odds(odds)
    print("Done.")


if __name__ == "__main__":
    main()
