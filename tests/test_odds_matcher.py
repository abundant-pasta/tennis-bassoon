from __future__ import annotations

import pandas as pd

from src.pipeline.tennis_odds_matcher import match_schedule_to_odds


def test_odds_matcher_rejects_ambiguous_candidates():
    schedule = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "match_date": "2026-04-28",
                "tourney_name": "Doha",
                "surface": "Hard",
                "round": "R32",
                "player_name": "Alpha One",
                "opp_name": "Beta Two",
                "player_rank": 10,
                "opp_rank": 20,
                "best_of": 3,
                "draw_size": 32,
            }
        ]
    )
    odds = pd.DataFrame(
        [
            {
                "match_date": "2026-04-28",
                "tourney_name": "Doha",
                "surface": "Hard",
                "round": "R32",
                "player_name": "Alpha One",
                "opp_name": "Beta Two",
                "player_decimal_odds": 1.80,
                "opp_decimal_odds": 2.10,
            },
            {
                "match_date": "2026-04-28",
                "tourney_name": "Doha",
                "surface": "Hard",
                "round": "R32",
                "player_name": "Beta Two",
                "opp_name": "Alpha One",
                "player_decimal_odds": 2.10,
                "opp_decimal_odds": 1.80,
            },
        ]
    )

    result = match_schedule_to_odds(schedule, odds)
    assert result.matched.empty
    assert list(result.rejected["rejection_reason"].unique()) == ["ambiguous_odds_match"]


def test_odds_matcher_reorients_reversed_odds_pair():
    schedule = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "match_date": "2026-05-17",
                "tourney_name": "Italian Open",
                "surface": "Clay",
                "round": "F",
                "player_name": "Jannik Sinner",
                "opp_name": "Casper Ruud",
                "player_rank": 1,
                "opp_rank": 25,
                "best_of": 3,
                "draw_size": 32,
            }
        ]
    )
    odds = pd.DataFrame(
        [
            {
                "match_date": "2026-05-17",
                "tourney_name": "Italian Open",
                "surface": "Clay",
                "round": "F",
                "player_name": "Casper Ruud",
                "opp_name": "Jannik Sinner",
                "player_decimal_odds": 5.75,
                "opp_decimal_odds": 1.15,
            }
        ]
    )

    result = match_schedule_to_odds(schedule, odds)

    assert result.rejected.empty
    row = result.matched.iloc[0]
    assert row["player_name"] == "Jannik Sinner"
    assert row["opp_name"] == "Casper Ruud"
    assert row["player_decimal_odds"] == 1.15
    assert row["opp_decimal_odds"] == 5.75
    assert row["novig_player_prob"] > row["novig_opp_prob"]
