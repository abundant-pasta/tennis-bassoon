"""The Odds API adapter for current ATP moneyline odds."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from statistics import median
import urllib.parse
import urllib.request

import pandas as pd


def _norm(text: str) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


@dataclass(frozen=True)
class TournamentMeta:
    sport_key: str
    title: str
    surface: str
    aliases: tuple[str, ...]


TOURNAMENTS: tuple[TournamentMeta, ...] = (
    TournamentMeta("tennis_atp_aus_open_singles", "Australian Open", "Hard", ("australian open", "atp australian open")),
    TournamentMeta("tennis_atp_barcelona_open", "Barcelona Open", "Clay", ("barcelona open", "atp barcelona open", "barcelona open banc sabadell")),
    TournamentMeta("tennis_atp_canadian_open", "Canadian Open", "Hard", ("canadian open", "atp canadian open", "national bank open", "rogers cup")),
    TournamentMeta("tennis_atp_china_open", "China Open", "Hard", ("china open", "atp china open", "beijing open")),
    TournamentMeta("tennis_atp_cincinnati_open", "Cincinnati Open", "Hard", ("cincinnati open", "atp cincinnati open", "western southern open")),
    TournamentMeta("tennis_atp_dubai", "Dubai Championships", "Hard", ("dubai championships", "atp dubai", "dubai duty free tennis championships")),
    TournamentMeta("tennis_atp_french_open", "French Open", "Clay", ("french open", "roland garros", "atp french open")),
    TournamentMeta("tennis_atp_indian_wells", "Indian Wells", "Hard", ("indian wells", "bnp paribas open", "atp indian wells")),
    TournamentMeta("tennis_atp_italian_open", "Italian Open", "Clay", ("italian open", "internazionali bnl d'italia", "atp italian open", "rome masters")),
    TournamentMeta("tennis_atp_madrid_open", "Madrid Open", "Clay", ("madrid open", "mutua madrid open", "atp madrid open")),
    TournamentMeta("tennis_atp_miami_open", "Miami Open", "Hard", ("miami open", "miami masters", "atp miami open")),
    TournamentMeta("tennis_atp_monte_carlo_masters", "Monte-Carlo Masters", "Clay", ("monte-carlo masters", "monte carlo masters", "rolex monte-carlo masters", "atp monte-carlo masters")),
    TournamentMeta("tennis_atp_munich", "Munich", "Clay", ("munich", "bmw open", "atp munich")),
    TournamentMeta("tennis_atp_paris_masters", "Paris Masters", "Hard", ("paris masters", "atp paris masters", "rolex paris masters")),
    TournamentMeta("tennis_atp_qatar_open", "Qatar Open", "Hard", ("qatar open", "atp qatar open", "qatar exxonmobil open", "doha open")),
    TournamentMeta("tennis_atp_shanghai_masters", "Shanghai Masters", "Hard", ("shanghai masters", "atp shanghai masters", "rolex shanghai masters")),
    TournamentMeta("tennis_atp_us_open", "US Open", "Hard", ("us open", "atp us open")),
    TournamentMeta("tennis_atp_wimbledon", "Wimbledon", "Grass", ("wimbledon", "atp wimbledon", "the championships wimbledon")),
)

_TOURNAMENT_ALIAS_LOOKUP = {
    _norm(alias): meta for meta in TOURNAMENTS for alias in meta.aliases + (meta.title, meta.sport_key)
}
_TOURNAMENT_KEY_LOOKUP = {meta.sport_key: meta for meta in TOURNAMENTS}


def resolve_tournaments(schedule_df: pd.DataFrame, override_sport_keys: tuple[str, ...] = ()) -> tuple[list[TournamentMeta], list[str]]:
    if override_sport_keys:
        resolved = []
        for sport_key in override_sport_keys:
            meta = _TOURNAMENT_KEY_LOOKUP.get(sport_key)
            if meta is None:
                meta = TournamentMeta(sport_key=sport_key, title=sport_key, surface="", aliases=(sport_key,))
            resolved.append(meta)
        return resolved, []

    resolved: list[TournamentMeta] = []
    unsupported: list[str] = []
    seen: set[str] = set()
    for raw_name in schedule_df.get("tourney_name", pd.Series(dtype="object")).dropna().astype(str).unique():
        key = _norm(raw_name)
        meta = _TOURNAMENT_ALIAS_LOOKUP.get(key)
        if meta is None:
            unsupported.append(raw_name)
            continue
        if meta.sport_key in seen:
            continue
        seen.add(meta.sport_key)
        resolved.append(meta)
    return resolved, unsupported


def fetch_tennis_odds_consensus(
    *,
    api_key: str,
    schedule_df: pd.DataFrame,
    api_base: str,
    regions: str,
    bookmakers: str | None,
    markets: str,
    odds_format: str,
    override_sport_keys: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, dict]:
    tournaments, unsupported = resolve_tournaments(schedule_df, override_sport_keys=override_sport_keys)
    payloads: list[dict] = []
    rows: list[dict] = []
    request_headers: list[dict] = []

    for meta in tournaments:
        params = {
            "apiKey": api_key,
            "markets": markets,
            "oddsFormat": odds_format,
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        else:
            params["regions"] = regions

        url = f"{api_base}/sports/{meta.sport_key}/odds?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "tennis-shadow-runner/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw_payload = resp.read()
            data = json.loads(raw_payload.decode("utf-8"))
            payloads.append({"sport_key": meta.sport_key, "response": data})
            request_headers.append(
                {
                    "sport_key": meta.sport_key,
                    "x_requests_remaining": resp.headers.get("x-requests-remaining"),
                    "x_requests_used": resp.headers.get("x-requests-used"),
                    "x_requests_last": resp.headers.get("x-requests-last"),
                }
            )

        for event in data:
            event_id = event.get("id", "")
            event_time = event.get("commence_time")
            player_name = event.get("home_team", "")
            opp_name = event.get("away_team", "")
            player_prices: list[float] = []
            opp_prices: list[float] = []
            bookmaker_keys: list[str] = []
            last_updates: list[str] = []

            for bookmaker_row in event.get("bookmakers", []):
                bk_key = bookmaker_row.get("key", "")
                for market in bookmaker_row.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    player_price = None
                    opp_price = None
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name", "")
                        price = outcome.get("price")
                        if name == player_name and price is not None:
                            player_price = float(price)
                        elif name == opp_name and price is not None:
                            opp_price = float(price)
                    if player_price is None or opp_price is None:
                        continue
                    player_prices.append(player_price)
                    opp_prices.append(opp_price)
                    bookmaker_keys.append(bk_key)
                    last_updates.append(str(market.get("last_update") or bookmaker_row.get("last_update") or ""))

            if not player_prices or not opp_prices:
                continue

            rows.append(
                {
                    "match_id": f"oddsapi_{event_id}",
                    "match_date": event_time,
                    "tourney_name": meta.title,
                    "surface": meta.surface,
                    "round": "",
                    "player_name": player_name,
                    "opp_name": opp_name,
                    "player_decimal_odds": round(float(median(player_prices)), 4),
                    "opp_decimal_odds": round(float(median(opp_prices)), 4),
                    "provider": "odds_api_consensus",
                    "sport_key": meta.sport_key,
                    "bookmaker_count": len(bookmaker_keys),
                    "bookmaker_keys": ",".join(sorted(set(bookmaker_keys))),
                    "odds_last_update": max(last_updates) if last_updates else None,
                }
            )

    frame = pd.DataFrame(rows)
    raw_payload = json.dumps(
        {
            "captured_utc": datetime.now(timezone.utc).isoformat(),
            "requests": payloads,
            "request_headers": request_headers,
            "unsupported_tournaments": unsupported,
        }
    ).encode("utf-8")
    meta = {
        "raw_payload": raw_payload,
        "request_headers": request_headers,
        "unsupported_tournaments": unsupported,
        "requested_tournaments": [meta.sport_key for meta in tournaments],
    }
    return frame, meta
