"""
アルゴリズムレジストリ

利用可能なMLアルゴリズムの一覧管理とファクトリーパターンを提供します。
新しいアルゴリズムを追加した際は、このファイルに登録してください。
"""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class AlgorithmType(Enum):
    """アルゴリズムタイプの列挙"""

    TREE_BASED = "tree_based"
    LINEAR = "linear"
    ENSEMBLE = "ensemble"
    BOOSTING = "boosting"
    PROBABILISTIC = "probabilistic"
    INSTANCE_BASED = "instance_based"
    NEURAL_NETWORK = "neural_network"


class AlgorithmCapability(Enum):
    """アルゴリズムの機能"""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    PROBABILITY_PREDICTION = "probability_prediction"
    FEATURE_IMPORTANCE = "feature_importance"
    INCREMENTAL_LEARNING = "incremental_learning"
    MULTICLASS = "multiclass"


class AlgorithmRegistry:
    """
    アルゴリズムレジストリクラス

    利用可能なMLアルゴリズムの管理とインスタンス化を行います。
    """

    def __init__(self):
        """初期化"""
        self._algorithms = {}
        self._register_algorithms()

    def _register_algorithms(self):
        """利用可能なアルゴリズムを登録"""

        # 既存のアルゴリズム
        self._algorithms.update(
            {
                # ツリー系
                "randomforest": {
                    "class_name": "RandomForestModel",
                    "module_path": "app.services.ml.models.randomforest_wrapper",
                    "type": AlgorithmType.TREE_BASED,
                    "capabilities": [
                        AlgorithmCapability.CLASSIFICATION,
                        AlgorithmCapability.PROBABILITY_PREDICTION,
                        AlgorithmCapability.FEATURE_IMPORTANCE,
                        AlgorithmCapability.MULTICLASS,
                    ],
                    "description": "ランダムフォレスト - 複数の決定木のアンサンブル",
                    "pros": ["高い精度", "特徴量重要度", "オーバーフィッティング耐性"],
                    "cons": ["解釈性が低い", "メモリ使用量大"],
                    "best_for": [
                        "中規模データ",
                        "ノイズ耐性が必要",
                        "特徴量重要度が必要",
                    ],
                },
                "extratrees": {
                    "class_name": "ExtraTreesModel",
                    "module_path": "app.services.ml.models.extratrees_wrapper",
                    "type": AlgorithmType.TREE_BASED,
                    "capabilities": [
                        AlgorithmCapability.CLASSIFICATION,
                        AlgorithmCapability.PROBABILITY_PREDICTION,
                        AlgorithmCapability.FEATURE_IMPORTANCE,
                        AlgorithmCapability.MULTICLASS,
                    ],
                    "description": "エクストラツリー - より高いランダム性を持つ決定木アンサンブル",
                    "pros": ["高速学習", "オーバーフィッティング耐性", "高い汎化性能"],
                    "cons": ["解釈性が低い", "ハイパーパラメータ調整が重要"],
                    "best_for": [
                        "大規模データ",
                        "高速学習が必要",
                        "ノイズの多いデータ",
                    ],
                },
                # ブースティング系
                "gradientboosting": {
                    "class_name": "GradientBoostingModel",
                    "module_path": "app.services.ml.models.gradientboosting_wrapper",
                    "type": AlgorithmType.BOOSTING,
                    "capabilities": [
                        AlgorithmCapability.CLASSIFICATION,
                        AlgorithmCapability.PROBABILITY_PREDICTION,
                        AlgorithmCapability.FEATURE_IMPORTANCE,
                        AlgorithmCapability.MULTICLASS,
                    ],
                    "description": "グラディエントブースティング - 逐次的に弱学習器を改善",
                    "pros": ["高い精度", "特徴量重要度", "柔軟性"],
                    "cons": ["オーバーフィッティングしやすい", "学習時間長"],
                    "best_for": [
                        "高精度が必要",
                        "構造化データ",
                        "特徴量エンジニアリング済み",
                    ],
                },
                "adaboost": {
                    "class_name": "AdaBoostModel",
                    "module_path": "app.services.ml.models.adaboost_wrapper",
                    "type": AlgorithmType.BOOSTING,
                    "capabilities": [
                        AlgorithmCapability.CLASSIFICATION,
                        AlgorithmCapability.PROBABILITY_PREDICTION,
                        AlgorithmCapability.FEATURE_IMPORTANCE,
                        AlgorithmCapability.MULTICLASS,
                    ],
                    "description": "アダブースト - 適応的ブースティング",
                    "pros": ["シンプル", "解釈しやすい", "少ないハイパーパラメータ"],
                    "cons": ["ノイズに敏感", "外れ値に弱い"],
                    "best_for": [
                        "クリーンなデータ",
                        "シンプルなモデルが必要",
                        "二値分類",
                    ],
                },
                # 線形系
                "ridge": {
                    "class_name": "RidgeModel",
                    "module_path": "app.services.ml.models.ridge_wrapper",
                    "type": AlgorithmType.LINEAR,
                    "capabilities": [
                        AlgorithmCapability.CLASSIFICATION,
                        AlgorithmCapability.FEATURE_IMPORTANCE,
                        AlgorithmCapability.MULTICLASS,
                    ],
                    "description": "リッジ分類器 - L2正則化線形分類器",
                    "pros": ["高速", "解釈しやすい", "正則化効果"],
                    "cons": ["確率予測なし", "非線形関係を捉えられない"],
                    "best_for": ["線形関係", "高次元データ", "高速予測が必要"],
                    "note": "predict_probaメソッドなし",
                },
                # 確率的
                "naivebayes": {
                    "class_name": "NaiveBayesModel",
                    "module_path": "app.services.ml.models.naivebayes_wrapper",
                    "type": AlgorithmType.PROBABILISTIC,
                    "capabilities": [
                        AlgorithmCapability.CLASSIFICATION,
                        AlgorithmCapability.PROBABILITY_PREDICTION,
                        AlgorithmCapability.MULTICLASS,
                        AlgorithmCapability.INCREMENTAL_LEARNING,
                    ],
                    "description": "ナイーブベイズ - ベイズの定理に基づく確率的分類器",
                    "pros": ["高速", "少ないデータでも動作", "確率的解釈"],
                    "cons": ["特徴量独立性の仮定", "連続値に制限"],
                    "best_for": ["テキスト分類", "小規模データ", "高速学習が必要"],
                },
                # インスタンスベース
                "knn": {
                    "class_name": "KNNModel",
                    "module_path": "app.services.ml.models.knn_wrapper",
                    "type": AlgorithmType.INSTANCE_BASED,
                    "capabilities": [
                        AlgorithmCapability.CLASSIFICATION,
                        AlgorithmCapability.PROBABILITY_PREDICTION,
                        AlgorithmCapability.MULTICLASS,
                    ],
                    "description": "K近傍法 - 近傍インスタンスに基づく分類",
                    "pros": ["シンプル", "非線形関係対応", "局所的パターン検出"],
                    "cons": ["計算コスト高", "メモリ使用量大", "次元の呪い"],
                    "best_for": ["小規模データ", "局所的パターン", "非線形関係"],
                },
            }
        )

    def get_available_algorithms(self) -> List[str]:
        """
        利用可能なアルゴリズム名のリストを取得

        Returns:
            アルゴリズム名のリスト
        """
        return list(self._algorithms.keys())

    def get_algorithm_info(self, algorithm_name: str) -> Optional[Dict[str, Any]]:
        """
        指定されたアルゴリズムの情報を取得

        Args:
            algorithm_name: アルゴリズム名

        Returns:
            アルゴリズム情報の辞書、存在しない場合はNone
        """
        return self._algorithms.get(algorithm_name)

    def get_algorithms_by_type(self, algorithm_type: AlgorithmType) -> List[str]:
        """
        指定されたタイプのアルゴリズムを取得

        Args:
            algorithm_type: アルゴリズムタイプ

        Returns:
            該当するアルゴリズム名のリスト
        """
        return [
            name
            for name, info in self._algorithms.items()
            if info["type"] == algorithm_type
        ]

    def get_algorithms_by_capability(
        self, capability: AlgorithmCapability
    ) -> List[str]:
        """
        指定された機能を持つアルゴリズムを取得

        Args:
            capability: 必要な機能

        Returns:
            該当するアルゴリズム名のリスト
        """
        return [
            name
            for name, info in self._algorithms.items()
            if capability in info["capabilities"]
        ]

    def create_algorithm_instance(
        self, algorithm_name: str, automl_config: Optional[Dict[str, Any]] = None
    ):
        """
        指定されたアルゴリズムのインスタンスを作成

        Args:
            algorithm_name: アルゴリズム名
            automl_config: AutoML設定

        Returns:
            アルゴリズムのインスタンス

        Raises:
            ValueError: 存在しないアルゴリズム名が指定された場合
            ImportError: モジュールのインポートに失敗した場合
        """
        if algorithm_name not in self._algorithms:
            available = ", ".join(self.get_available_algorithms())
            raise ValueError(
                f"未知のアルゴリズム: {algorithm_name}. 利用可能: {available}"
            )

        algorithm_info = self._algorithms[algorithm_name]

        try:
            # モジュールを動的インポート
            module_path = algorithm_info["module_path"]
            class_name = algorithm_info["class_name"]

            module = __import__(module_path, fromlist=[class_name])
            algorithm_class = getattr(module, class_name)

            # インスタンス作成
            instance = algorithm_class(automl_config=automl_config)

            logger.info(f"✅ {algorithm_name}アルゴリズムのインスタンスを作成")
            return instance

        except ImportError as e:
            logger.error(f"❌ {algorithm_name}のインポートエラー: {e}")
            raise ImportError(f"アルゴリズム {algorithm_name} のインポートに失敗: {e}")
        except AttributeError as e:
            logger.error(f"❌ {algorithm_name}のクラス取得エラー: {e}")
            raise ImportError(
                f"アルゴリズム {algorithm_name} のクラス {class_name} が見つかりません: {e}"
            )

    def get_algorithm_summary(self) -> Dict[str, Any]:
        """
        全アルゴリズムのサマリーを取得

        Returns:
            アルゴリズムサマリーの辞書
        """
        summary = {
            "total_algorithms": len(self._algorithms),
            "by_type": {},
            "by_capability": {},
            "algorithms": {},
        }

        # タイプ別集計
        for algorithm_type in AlgorithmType:
            algorithms = self.get_algorithms_by_type(algorithm_type)
            if algorithms:
                summary["by_type"][algorithm_type.value] = algorithms

        # 機能別集計
        for capability in AlgorithmCapability:
            algorithms = self.get_algorithms_by_capability(capability)
            if algorithms:
                summary["by_capability"][capability.value] = algorithms

        # 各アルゴリズムの基本情報
        for name, info in self._algorithms.items():
            summary["algorithms"][name] = {
                "type": info["type"].value,
                "description": info["description"],
                "capabilities": [cap.value for cap in info["capabilities"]],
            }

        return summary

    def print_algorithm_catalog(self):
        """アルゴリズムカタログを出力"""

        for algorithm_type in AlgorithmType:
            algorithms = self.get_algorithms_by_type(algorithm_type)
            if algorithms:
                print(f"\n📊 {algorithm_type.value.upper().replace('_', ' ')}:")
                for algo_name in algorithms:
                    info = self._algorithms[algo_name]
                    print(f"  • {algo_name}: {info['description']}")
                    print(f"    長所: {', '.join(info['pros'])}")
                    print(f"    適用場面: {', '.join(info['best_for'])}")
                    if "note" in info:
                        print(f"    注意: {info['note']}")


# グローバルインスタンス
algorithm_registry = AlgorithmRegistry()


if __name__ == "__main__":
    # カタログ表示のテスト
    registry = AlgorithmRegistry()
    registry.print_algorithm_catalog()

    # サマリー表示
    summary = registry.get_algorithm_summary()
    print(f"\n📈 総アルゴリズム数: {summary['total_algorithms']}")
    print(
        f"確率予測対応: {len(summary['by_capability'].get('probability_prediction', []))}個"
    )
    print(
        f"特徴量重要度対応: {len(summary['by_capability'].get('feature_importance', []))}個"
    )
