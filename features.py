"""
各种足球数据的特征工程
需要注意的是每个数据来源的坐标不一样，具体可见
https://mplsoccer.readthedocs.io/en/latest/gallery/pitch_setup/plot_compare_pitches.html
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class BasicXG:
    """最基本的 xG 计算，通过射门角度和距离计算 xG"""

    def __init__(self, shots: pd.DataFrame) -> None:
        self.data = shots[["id", "player_name"]].copy()

        # 一些计算需要的基本数值
        self.data["goal"] = shots["outcome_name"].apply(
            lambda cell: 1 if cell == "Goal" else 0
        )
        self.data["x"] = shots["x"].apply(lambda x: 105 - x)
        self.data["c"] = shots["y"].apply(lambda x: abs(34 - x))
        self.data["distance"] = np.sqrt(
            self.data["x"] ** 2 + self.data["c"] ** 2
        )
        self.data["angle"] = self._calculate_angle()
        self.data["basic_xg"] = self._train()

        self.features = [
            "id",
            "player_name",
            "goal",
            "angle",
            "distance",
            "basic_xg",
        ]

    def _calculate_angle(self) -> np.ndarray:
        angle = (
            np.where(
                np.arctan(
                    7.32
                    * self.data["x"]
                    / (
                        self.data["x"] ** 2
                        + self.data["c"] ** 2
                        - (7.32 / 2) ** 2
                    )
                )
                >= 0,
                np.arctan(
                    7.32
                    * self.data["x"]
                    / (
                        self.data["x"] ** 2
                        + self.data["c"] ** 2
                        - (7.32 / 2) ** 2
                    )
                ),
                np.arctan(
                    7.32
                    * self.data["x"]
                    / (
                        self.data["x"] ** 2
                        + self.data["c"] ** 2
                        - (7.32 / 2) ** 2
                    )
                )
                + np.pi,
            )
            * 180
            / np.pi
        )
        return angle

    def _train(self) -> np.ndarray:
        # 使用射门距离和角度计算基础的 xg
        X = self.data[["angle", "distance"]]
        y = self.data["goal"]
        model = LogisticRegression(random_state=42).fit(X, y)
        xg = model.predict_proba(X)[:, 1]
        return xg

    def evaluate(self) -> dict:
        """使用准确度、精确度、召回率和 AUC 指标评估逻辑回归模型"""
        y_pred = self.data["basic_xg"] >= 0.5
        y = self.data["goal"]
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        auc = roc_auc_score(y, y_pred)
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc": auc,
        }
        return metrics

    def to_df(self) -> pd.DataFrame:
        df = self.data[self.features]
        return df


class StatsbombXG(BasicXG):
    """Statsbomb 的预期进球特征
    https://twitter.com/StatsBomb/status/1650847925197471745
    * 射门距离
    * 射门角度
    * 基础 xg
    * 守门员的位置
    * 守门员与射门的距离
    * 射门是否比守门员更接近球门
    * 干扰射门的防守球员(球半径 1 米内的防守球员)
    * 射门与球门形成的三角形内防守球员
    * 射门与球门形成的三角形内进攻球员
    * 射门时球的高度(身体位置相对于地面高度)
    """

    def __init__(self, shots: pd.DataFrame, tracks: pd.DataFrame) -> None:
        """tracks: 计算标准的 xG 需要用到队员和对手的追踪数据"""
        super().__init__(shots)

        # 过滤射门的追踪事件并把射门合并进来
        opponents = tracks.loc[tracks["id"].isin(shots["id"])].loc[
            tracks["teammate"] == False
        ]
        self.opponents_tracks = opponents.merge(
            shots, on="id", how="outer", suffixes=("", "_shot")
        )
        teammate = tracks.loc[tracks["id"].isin(shots["id"])].loc[
            tracks["teammate"] == True
        ]
        self.teammate_tracks = teammate.merge(
            shots, on="id", how="outer", suffixes=("", "_shot")
        )
        # 把射门的守门员事件合并进来
        gks = opponents.loc[opponents["position_id"] == 1]
        self.shots = shots.merge(
            gks,
            on="id",
            how="inner",
            suffixes=("", "_gk"),
        )

        self.data[["x_gk", "y_gk"]] = self.shots[["x_gk", "y_gk"]]
        self.data["to_gk_dist"] = self._calculate_shot_to_gk_distance()
        self.data["goal_to_gk_dist"] = self._calculate_goal_to_gk_distance()
        self.data["is_closer"] = np.where(
            self.data["goal_to_gk_dist"] > self.data["distance"], 1, 0
        )
        self.data = self.data.merge(self._opponents_in_interference(), on="id")
        self.data = self.data.merge(
            self._players_in_triangle(
                self.opponents_tracks, "opponents_triangle"
            ),
            on="id",
            how="outer",
        ).fillna(0)
        self.data = self.data.merge(
            self._players_in_triangle(
                self.teammate_tracks, "teammate_triangle"
            ),
            on="id",
            how="outer",
        ).fillna(0)
        self.data["height"] = shots["body_part_name"].apply(
            self._height_furmula
        )

        self.features += [
            "x_gk",
            "y_gk",
            "to_gk_dist",
            "is_closer",
            "close_players",
            "opponents_triangle",
            "teammate_triangle",
            "height",
        ]

    def _calculate_shot_to_gk_distance(self) -> pd.Series:
        distance = np.sqrt(
            (self.shots["x"] - self.shots["x_gk"]) ** 2
            + (self.shots["y"] - self.shots["y_gk"]) ** 2
        )
        return distance

    def _calculate_goal_to_gk_distance(self) -> pd.Series:
        distance = np.sqrt(
            (105 - self.shots["x_gk"]) ** 2 + (34 - self.shots["y_gk"]) ** 2
        )
        return distance

    def _opponents_in_interference(self) -> pd.Series:
        # 计算射门与追踪事件的距离
        self.opponents_tracks["distance"] = np.sqrt(
            (self.opponents_tracks["x_shot"] - self.opponents_tracks["x"]) ** 2
            + (self.opponents_tracks["y_shot"] - self.opponents_tracks["y"])
            ** 2
        )
        # 分组并计算距离小于 1 的数量
        close_players = self.opponents_tracks.groupby("id")["distance"].apply(
            lambda x: (x < 1).sum()
        )
        close_players.name = "close_players"
        return close_players

    def _players_in_triangle(
        self, tracks: pd.DataFrame, name: str
    ) -> pd.Series:
        # 过滤掉门将
        tracks = tracks.loc[tracks["position_id"] != 1]
        x1 = 105
        y1 = 34 - 7.32 / 2
        x2 = 105
        y2 = 34 + 7.32 / 2
        x3 = tracks["x_shot"]
        y3 = tracks["y_shot"]
        xp = tracks["x"]
        yp = tracks["y"]
        c1 = (x2 - x1) * (yp - y1) - (y2 - y1) * (xp - x1)
        c2 = (x3 - x2) * (yp - y2) - (y3 - y2) * (xp - x2)
        c3 = (x1 - x3) * (yp - y3) - (y1 - y3) * (xp - x3)
        tracks = tracks.loc[
            ((c1 < 0) & (c2 < 0) & (c3 < 0)) | ((c1 > 0) & (c2 > 0) & (c3 > 0))
        ]
        triangle = tracks.groupby("id")["id"].count()
        triangle.name = name
        return triangle

    def _height_furmula(self, body_part_name: str) -> float:
        """射门时球的高度
        Statsbomb 的射门部位只有 ['Right Foot', 'Head', 'Left Foot', 'Other'] 四种
        所以暂时人为的设置一个身高 1.8 米
        脚射门高度为 0.01
        其他部位射门高度为 1.0
        头球射门高度为 1.8
        """
        if body_part_name == "Head":
            return 1.8
        elif body_part_name == "Other":
            return 1.0
        else:
            return 0.01
