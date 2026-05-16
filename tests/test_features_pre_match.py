from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.tennis_engineer import compute_elo_ratings, compute_h2h


def test_feature_builders_allow_future_unsolved_rows():
    df = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "player_id": 1,
                "opp_id": 2,
                "player_name": "Alpha One",
                "opp_name": "Beta Two",
                "surface": "Hard",
                "round": "R32",
                "tourney_date": pd.Timestamp("2026-01-01"),
                "sim_date": pd.Timestamp("2026-01-01"),
                "won": 1,
                "score": "6-4 6-4",
                "best_of": 3,
            },
            {
                "match_id": "m2",
                "player_id": 1,
                "opp_id": 2,
                "player_name": "Alpha One",
                "opp_name": "Beta Two",
                "surface": "Hard",
                "round": "R16",
                "tourney_date": pd.Timestamp("2026-01-05"),
                "sim_date": pd.Timestamp("2026-01-05"),
                "won": np.nan,
                "score": "",
                "best_of": 3,
            },
        ]
    )

    elo_df = compute_elo_ratings(df)
    h2h_df = compute_h2h(df)

    assert elo_df.loc[1, "player_elo_overall"] > 1500.0
    assert int(h2h_df.loc[1, "h2h_n"]) == 1
    assert float(h2h_df.loc[1, "h2h_win_pct_adj"]) > 0.5
