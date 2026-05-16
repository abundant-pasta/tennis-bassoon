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
