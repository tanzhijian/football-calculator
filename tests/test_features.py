import pandas as pd

from features import BasicXG, StatsbombXG


shots = pd.read_csv("tests/test_shots.csv")
tracks = pd.read_csv("tests/test_tracks.csv")


def test_basicxg():
    xg = BasicXG(shots)
    metrics = xg.evaluate()
    assert metrics["precision"] > 0
    df = xg.to_df()
    basic_xg = df.iloc[0]["basic_xg"]
    assert 0 < basic_xg < 1


def test_statsbombxg():
    xg = StatsbombXG(shots, tracks)
    df = xg.to_df()
    df_shape = df.shape
    assert df_shape[0] == shots.shape[0]
    assert df_shape[1] == 14
    shot = df.iloc[0]
    shot_to_gk_dist = shot["to_gk_dist"]
    assert shot_to_gk_dist > 0
    is_closer = shot["is_closer"]
    assert is_closer in [1, 0]
    close_players = shot["close_players"]
    assert 0 <= close_players <= 11
    opponents_triangle = shot["opponents_triangle"]
    assert 0 <= opponents_triangle <= 11
    teammate_triangle = shot["teammate_triangle"]
    assert 0 <= teammate_triangle <= 11
    height = shot["height"]
    assert height in [0.01, 1.0, 1.8]
