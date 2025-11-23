import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from typing import Dict, Optional

class StackingService:
    """
    Level-2 メタ学習器
    Ridge回帰 (NNLS: Non-negative Least Squares) を使用して
    各ベースモデルの予測値を最適に統合する。
    """
    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Ridge回帰の正則化パラメータ
        """
        # positive=True で係数を非負に制約 (NNLS相当)
        # fit_intercept=False にして、純粋なモデルの重み付けとして扱うことが多いが、
        # バイアス補正のためにTrueにすることもある。ここでは予測値の加重平均を目指すためFalse推奨だが、
        # ドキュメントの文脈（Ridge Regression）に従い、柔軟性を持たせる。
        # ただし、確率の重み付け平均という意味合いを強くするなら fit_intercept=False が自然。
        # ここでは切片なしで、純粋な重み付けを学習させる。
        self.model = Ridge(alpha=alpha, positive=True, fit_intercept=False, random_state=42)
        self.weights: Optional[Dict[str, float]] = None
        self.feature_names: list = []

    def train(self, X_meta: pd.DataFrame, y_true: np.ndarray) -> None:
        """
        メタモデルを学習する
        
        Args:
            X_meta: ベースモデルの予測値 (各列が1つのモデルの予測確率)
            y_true: 正解ラベル
        """
        self.feature_names = X_meta.columns.tolist()
        
        # 学習
        self.model.fit(X_meta, y_true)
        
        # 重みの保存
        self.weights = dict(zip(self.feature_names, self.model.coef_))
        
        # 重みの合計で正規化して、合計が1になるように調整（解釈性のため）
        # ただし予測値には生モデルを使う（合計が1にならなくても確率として妥当なら良いが、
        # 確率の重み付け和にするなら正規化が必要）
        # ここではRidgeの出力をそのまま使う（キャリブレーション効果も期待）
        # ただし、出力が[0,1]を超えないようにpredictでクリップする

    def predict(self, X_meta: pd.DataFrame) -> np.ndarray:
        """
        予測を行う
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
            
        y_pred = self.model.predict(X_meta)
        
        # 確率なので [0, 1] にクリップ
        return np.clip(y_pred, 0.0, 1.0)

    def get_weights(self) -> Dict[str, float]:
        """学習された重みを取得"""
        if self.weights is None:
            return {}
        return self.weights
