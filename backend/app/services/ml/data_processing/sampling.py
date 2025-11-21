from typing import Tuple, Union
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN


class ImbalanceSampler:
    """
    クラス不均衡データを処理するためのサンプリングクラス。
    SMOTEやADASYNなどのオーバーサンプリング手法を提供します。
    """

    def __init__(self, method: str = "smote", random_state: int = 42):
        """
        初期化

        Args:
            method: サンプリング手法 ("smote" or "adasyn")
            random_state: ランダムシード
        """
        self.method = method
        self.random_state = random_state

        if method == "smote":
            self.sampler = SMOTE(random_state=random_state)
        elif method == "adasyn":
            self.sampler = ADASYN(random_state=random_state)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def fit_resample(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
        """
        データセットをリサンプリングします。

        Args:
            X: 特徴量
            y: ラベル

        Returns:
            リサンプリングされたX, y
        """
        return self.sampler.fit_resample(X, y)
