"""
フィットネス共有（Fitness Sharing）

遺伝的アルゴリズムにおけるニッチ形成を実現するためのフィットネス共有機能。
類似した個体のフィットネス値を調整することで、多様な戦略の共存を促進します。
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from ..config.constants import OPERATORS
from ..genes import ConditionGroup, StrategyGene
from ..serializers.serialization import GeneSerializer
from ..utils.indicator_utils import get_valid_indicator_types

logger = logging.getLogger(__name__)


class FitnessSharing:
    """
    フィットネス共有クラス

    個体間の類似度を計算し、類似した個体のフィットネス値を調整することで
    多様な戦略の共存を促進します。
    """

    # 大規模集団でサンプリングを使用する閾値（デフォルト: 200個体以上）
    DEFAULT_SAMPLING_THRESHOLD = 200
    # サンプリング時に使用するサンプル数の割合
    SAMPLING_RATIO = 0.3

    def __init__(
        self,
        sharing_radius: float = 0.1,
        alpha: float = 1.0,
        sampling_threshold: Optional[int] = None,
        sampling_ratio: Optional[float] = None,
    ):
        """
        初期化

        Args:
            sharing_radius: 共有半径（類似度の閾値）
            alpha: 共有関数の形状パラメータ
            sampling_threshold: サンプリングを使用する集団サイズの閾値
            sampling_ratio: サンプリング時に使用するサンプル数の割合
        """
        self.sharing_radius = sharing_radius
        self.alpha = alpha
        self.gene_serializer = GeneSerializer()
        self.sampling_threshold = (
            sampling_threshold
            if sampling_threshold is not None
            else self.DEFAULT_SAMPLING_THRESHOLD
        )
        self.sampling_ratio = (
            sampling_ratio if sampling_ratio is not None else self.SAMPLING_RATIO
        )

        # 指標タイプマップの初期化（ベクトル化用）
        try:
            self.indicator_types = get_valid_indicator_types()
            # 安定した順序を保証するためにソート
            self.indicator_types.sort()
            self.indicator_map = {
                name: i for i, name in enumerate(self.indicator_types)
            }
        except Exception as e:
            logger.warning(f"指標タイプ取得失敗: {e}")
            self.indicator_types = []
            self.indicator_map = {}

        # オペレータマップの初期化
        try:
            # 比較演算子に加えて論理演算子も評価対象にする
            self.operator_types = OPERATORS.copy()
            self.operator_types.extend(["AND", "OR"])
            self.operator_types.sort()
            self.operator_map = {op: i for i, op in enumerate(self.operator_types)}
        except Exception as e:
            logger.warning(f"オペレータタイプ取得失敗: {e}")
            self.operator_types = []
            self.operator_map = {}

    def apply_fitness_sharing(self, population: List[Any]) -> List[Any]:
        """
        個体群にフィットネス共有を適用（最適化版）

        ベクトル化とKD-Treeを使用してO(N²)からO(N log N)に計算量を削減。

        Args:
            population: 個体群

        Returns:
            フィットネス共有適用後の個体群
        """
        try:
            if len(population) <= 1:
                return population

            # 各個体の戦略遺伝子を取得してベクトル化
            genes = []
            vectors = []
            valid_indices = []

            for i, individual in enumerate(population):
                try:
                    gene = self.gene_serializer.from_list(individual, StrategyGene)
                    if gene is not None:
                        genes.append(gene)
                        vectors.append(self._vectorize_gene(gene))
                        valid_indices.append(i)
                    else:
                        genes.append(None)
                except Exception as e:
                    logger.warning(f"個体のデコードに失敗: {e}")
                    genes.append(None)

            if len(vectors) < 2:
                return population

            vectors = np.array(vectors)

            # ベクトル次元数チェックとパディング（次元不一致対策）
            max_dim = max(v.shape[0] for v in vectors if isinstance(v, np.ndarray))
            vectors_padded = []
            for v in vectors:
                if v.shape[0] < max_dim:
                    padding = np.zeros(max_dim - v.shape[0])
                    vectors_padded.append(np.concatenate([v, padding]))
                else:
                    vectors_padded.append(v)
            vectors = np.array(vectors_padded)

            # 最適化されたニッチカウント計算
            niche_counts_vectorized = self.compute_niche_counts_vectorized(vectors)

            # 全個体用のニッチカウント配列を作成（デフォルト1.0）
            niche_counts = [1.0] * len(population)
            for idx, valid_idx in enumerate(valid_indices):
                niche_counts[valid_idx] = niche_counts_vectorized[idx]

            # フィットネス値を調整（多目的最適化対応）
            for i, individual in enumerate(population):
                if hasattr(individual, "fitness") and individual.fitness.valid:
                    original_fitness_values = individual.fitness.values
                    # 各目的関数のフィットネス値をニッチカウントで調整
                    shared_fitness_values = tuple(
                        fitness_val / niche_counts[i]
                        for fitness_val in original_fitness_values
                    )
                    individual.fitness.values = shared_fitness_values

            # ニッチ計算後にシルエットベース共有を適用
            return self.silhouette_based_sharing(population)

        except Exception as e:
            logger.error(f"フィットネス共有適用エラー: {e}")
            return population

    def compute_niche_counts_vectorized(self, vectors: np.ndarray) -> np.ndarray:
        """
        ベクトル化されたニッチカウント計算（O(N log N)）

        KD-Treeを使用して近傍探索を効率化し、大規模集団では
        サンプリングベースの近似を行う。

        Args:
            vectors: 個体の特徴ベクトル配列 (N x D)

        Returns:
            各個体のニッチカウント配列 (N,)
        """
        n_individuals = len(vectors)

        if n_individuals < 2:
            return np.ones(n_individuals)

        # ベクトルを正規化（距離計算の安定性向上）
        vectors_normalized = self._normalize_vectors(vectors)

        # 共有半径を距離空間に変換
        # 類似度 > (1 - sharing_radius) を共有範囲とする
        # 距離 < sharing_radius * max_distance として近似
        distance_threshold = self.sharing_radius * np.sqrt(vectors_normalized.shape[1])

        # 大規模集団の場合はサンプリングを使用
        if n_individuals > self.sampling_threshold:
            return self._compute_niche_counts_sampling(
                vectors_normalized, distance_threshold
            )

        # KD-Treeによる近傍探索
        neighbors_list = self.find_neighbors_kdtree(
            vectors_normalized, radius=distance_threshold
        )

        # ニッチカウントを計算
        niche_counts = np.array(
            [max(1.0, float(len(neighbors))) for neighbors in neighbors_list]
        )

        return niche_counts

    def find_neighbors_kdtree(
        self, vectors: np.ndarray, radius: float
    ) -> List[np.ndarray]:
        """
        KD-Treeを使用して各点の近傍を探索（O(N log N)）

        Args:
            vectors: 特徴ベクトル配列 (N x D)
            radius: 探索半径

        Returns:
            各点の近傍インデックスのリスト
        """
        if len(vectors) < 1:
            return []

        # KD-Treeを構築
        tree = cKDTree(vectors)

        # 各点の近傍を検索
        neighbors_list = tree.query_ball_point(vectors, r=radius)

        return neighbors_list

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        特徴ベクトルを正規化（0-1スケーリング）

        Args:
            vectors: 特徴ベクトル配列 (N x D)

        Returns:
            正規化されたベクトル配列
        """
        if len(vectors) == 0:
            return vectors

        # 各次元の最小・最大値を計算
        min_vals = vectors.min(axis=0)
        max_vals = vectors.max(axis=0)

        # ゼロ除算を防ぐ
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0

        # 正規化
        normalized = (vectors - min_vals) / range_vals

        return normalized

    def _compute_niche_counts_sampling(
        self, vectors: np.ndarray, distance_threshold: float
    ) -> np.ndarray:
        """
        サンプリングベースのニッチカウント近似（大規模集団用）

        全ペア計算の代わりに、ランダムサンプルとの距離を計算して
        ニッチカウントを推定する。

        Args:
            vectors: 正規化された特徴ベクトル配列 (N x D)
            distance_threshold: 距離閾値

        Returns:
            推定されたニッチカウント配列 (N,)
        """
        n_individuals = len(vectors)
        sample_size = max(10, int(n_individuals * self.sampling_ratio))

        # ランダムサンプルを選択
        np.random.seed(42)  # 再現性のため
        sample_indices = np.random.choice(
            n_individuals, size=min(sample_size, n_individuals), replace=False
        )
        sample_vectors = vectors[sample_indices]

        # サンプルでKD-Treeを構築
        sample_tree = cKDTree(sample_vectors)

        # 各個体からサンプルへの距離を計算
        distances, _ = sample_tree.query(vectors, k=min(10, len(sample_indices)))

        # 閾値内のサンプル数をカウントし、全体にスケール
        if distances.ndim == 1:
            distances = distances.reshape(-1, 1)

        neighbors_in_sample = np.sum(distances < distance_threshold, axis=1)

        # サンプル比率でスケールアップして推定
        scale_factor = n_individuals / len(sample_indices)
        niche_counts = np.maximum(1.0, neighbors_in_sample * scale_factor)

        return niche_counts

    def _calculate_similarity(self, gene1: StrategyGene, gene2: StrategyGene) -> float:
        """
        2つの戦略遺伝子間の類似度を計算

        Args:
            gene1: 戦略遺伝子1
            gene2: 戦略遺伝子2

        Returns:
            類似度（0.0-1.0）
        """
        try:
            similarity_scores = []

            # 指標の類似度
            indicator_similarity = self._calculate_indicator_similarity(
                gene1.indicators, gene2.indicators
            )
            similarity_scores.append(indicator_similarity)

            # ロング条件の類似度
            long_similarity = self._calculate_condition_similarity(
                gene1.long_entry_conditions, gene2.long_entry_conditions
            )
            similarity_scores.append(long_similarity)

            # ショート条件の類似度
            short_similarity = self._calculate_condition_similarity(
                gene1.short_entry_conditions, gene2.short_entry_conditions
            )
            similarity_scores.append(short_similarity)

            # リスク管理の類似度
            risk_similarity = self._calculate_risk_management_similarity(
                gene1.risk_management, gene2.risk_management
            )
            similarity_scores.append(risk_similarity)

            # TP/SL遺伝子の類似度
            tpsl_similarity = self._calculate_tpsl_similarity(
                gene1.tpsl_gene, gene2.tpsl_gene
            )
            similarity_scores.append(tpsl_similarity)

            # ポジションサイジング遺伝子の類似度
            position_sizing_similarity = self._calculate_position_sizing_similarity(
                gene1.position_sizing_gene, gene2.position_sizing_gene
            )
            similarity_scores.append(position_sizing_similarity)

            # 重み付き平均（指標、条件、リスク管理、TP/SL、ポジションサイジングを考慮）
            # 各要素の重要度に応じて重みを調整
            weights = [0.2, 0.2, 0.2, 0.15, 0.15, 0.1]  # 合計1.0
            weighted_similarity = sum(
                score * weight for score, weight in zip(similarity_scores, weights)
            )

            return max(0.0, min(1.0, weighted_similarity))

        except Exception as e:
            logger.error(f"類似度計算エラー: {e}")
            return 0.0

    def _calculate_indicator_similarity(
        self, indicators1: List[Any], indicators2: List[Any]
    ) -> float:
        """指標の類似度を計算"""
        try:
            if not indicators1 or not indicators2:
                return 0.0 if indicators1 != indicators2 else 1.0

            # 指標タイプの集合を作成
            types1 = {ind.type for ind in indicators1}
            types2 = {ind.type for ind in indicators2}

            # Jaccard係数を計算
            intersection = len(types1 & types2)
            union = len(types1 | types2)

            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.0

    def _calculate_condition_similarity(
        self, conditions1: List[Any], conditions2: List[Any]
    ) -> float:
        """条件の類似度を計算"""
        try:
            if not conditions1 or not conditions2:
                return 0.0 if conditions1 != conditions2 else 1.0

            # 条件の構造的類似度を計算
            similar_conditions = 0
            total_conditions = max(len(conditions1), len(conditions2))

            for i in range(min(len(conditions1), len(conditions2))):
                cond1 = conditions1[i]
                cond2 = conditions2[i]

                # 演算子の一致
                if hasattr(cond1, "operator") and hasattr(cond2, "operator"):
                    if cond1.operator == cond2.operator:
                        similar_conditions += 0.5

                # オペランドタイプの一致（簡易版）
                if hasattr(cond1, "left_operand") and hasattr(cond2, "left_operand"):
                    if str(type(cond1.left_operand)) == str(type(cond2.left_operand)):
                        similar_conditions += 0.5

            return (
                similar_conditions / total_conditions if total_conditions > 0 else 0.0
            )

        except Exception:
            return 0.0

    def _calculate_risk_management_similarity(
        self, risk1: Dict[str, Any], risk2: Dict[str, Any]
    ) -> float:
        """リスク管理の類似度を計算"""
        try:
            if not risk1 or not risk2:
                return 0.0 if risk1 != risk2 else 1.0

            similarity_score = 0.0
            total_fields = 0

            # 共通フィールドの類似度を計算
            common_fields = set(risk1.keys()) & set(risk2.keys())

            for field in common_fields:
                total_fields += 1
                val1 = risk1[field]
                val2 = risk2[field]

                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # 数値の場合は相対差を計算
                    if val1 == 0 and val2 == 0:
                        similarity_score += 1.0
                    elif val1 != 0 or val2 != 0:
                        max_val = max(abs(val1), abs(val2))
                        if max_val > 0:
                            diff = abs(val1 - val2) / max_val
                            similarity_score += max(0.0, 1.0 - diff)
                else:
                    # その他の場合は完全一致
                    if val1 == val2:
                        similarity_score += 1.0

            return similarity_score / total_fields if total_fields > 0 else 0.0

        except Exception:
            return 0.0

    def _calculate_tpsl_similarity(self, tpsl1: Any, tpsl2: Any) -> float:
        """TP/SL遺伝子の類似度を計算"""
        if tpsl1 is None and tpsl2 is None:
            return 1.0
        if tpsl1 is None or tpsl2 is None:
            return 0.0

        # 例: メソッドと主要パラメータの一致度
        score = 0.0
        if tpsl1.method == tpsl2.method:
            score += 0.5

        # 数値パラメータの類似度（例: stop_loss_pct, take_profit_pct）
        if tpsl1.stop_loss_pct is not None and tpsl2.stop_loss_pct is not None:
            diff = abs(tpsl1.stop_loss_pct - tpsl2.stop_loss_pct)
            score += max(
                0.0,
                0.25 * (1 - diff / max(tpsl1.stop_loss_pct, tpsl2.stop_loss_pct, 1e-6)),
            )

        if tpsl1.take_profit_pct is not None and tpsl2.take_profit_pct is not None:
            diff = abs(tpsl1.take_profit_pct - tpsl2.take_profit_pct)
            score += max(
                0.0,
                0.25
                * (1 - diff / max(tpsl1.take_profit_pct, tpsl2.take_profit_pct, 1e-6)),
            )

        return min(1.0, score)

    def _calculate_position_sizing_similarity(self, ps1: Any, ps2: Any) -> float:
        """ポジションサイジング遺伝子の類似度を計算"""
        if ps1 is None and ps2 is None:
            return 1.0
        if ps1 is None or ps2 is None:
            return 0.0

        # 例: メソッドと主要パラメータの一致度
        score = 0.0
        if ps1.method == ps2.method:
            score += 0.5

        # 数値パラメータの類似度（例: risk_per_trade）
        if ps1.risk_per_trade is not None and ps2.risk_per_trade is not None:
            diff = abs(ps1.risk_per_trade - ps2.risk_per_trade)
            score += max(
                0.0,
                0.5 * (1 - diff / max(ps1.risk_per_trade, ps2.risk_per_trade, 1e-6)),
            )

        return min(1.0, score)

    def silhouette_based_sharing(self, population: List[Any]) -> List[Any]:
        """
        シルエットベースの共有

        KMeansクラスタリングとシルエットスコアを使って個体のフィットネスを調整します。
        クラスタ内での凝集度が高い個体ほどフィットネスが高くなります。

        Args:
            population: 個体群

        Returns:
            シルエットベース調整適用後の個体群
        """
        try:
            if len(population) <= 1:
                return population

            # 個体の特徴ベクトルを作成
            vectors = []
            valid_indices = []
            for i, individual in enumerate(population):
                try:
                    gene = self.gene_serializer.from_list(individual, StrategyGene)
                    if gene is not None:
                        vector = self._vectorize_gene(gene)
                        vectors.append(vector)
                        valid_indices.append(i)
                except Exception:
                    continue

            if len(vectors) <= 1:
                return population

            vectors = np.array(vectors)
            n_clusters = min(len(vectors), 3)  # 最大3クラスタ

            # KMeansクラスタリング
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(vectors)

            # シルエットスコア計算
            silhouette_vals = silhouette_samples(vectors, labels)

            # フィットネス調整
            for j, idx in enumerate(valid_indices):
                individual = population[idx]
                if hasattr(individual, "fitness") and individual.fitness.valid:
                    silhouette_score = silhouette_vals[j]
                    # シルエットスコアを0-1に正規化（-1 to 1 -> 0 to 1）
                    normalized_silhouette = (silhouette_score + 1.0) / 2.0
                    # 良いシルエットスコアほどfitnessを高く（調整係数小さく）
                    adjustment_factor = max(0.1, 1.0 - normalized_silhouette)

                    original_fitness_values = individual.fitness.values
                    adjusted_values = tuple(
                        fitness_val * adjustment_factor
                        for fitness_val in original_fitness_values
                    )
                    individual.fitness.values = adjusted_values

            return population

        except Exception as e:
            logger.error(f"シルエットベース共有エラー: {e}")
            return population

    def _vectorize_gene(self, gene: StrategyGene) -> np.ndarray:
        """
        StrategyGeneを特徴ベクトルに変換

        Args:
            gene: 戦略遺伝子

        Returns:
            特徴ベクトル
        """
        features = []

        # 指標数
        features.append(float(len(gene.indicators)))

        # 条件数
        features.append(float(len(gene.long_entry_conditions)))
        features.append(float(len(gene.short_entry_conditions)))

        # リスク管理パラメータ
        if gene.risk_management:
            features.append(float(gene.risk_management.get("position_size", 0.1)))
        else:
            features.append(0.1)

        # TP/SLパラメータ
        if gene.tpsl_gene:
            features.append(float(gene.tpsl_gene.stop_loss_pct or 0.05))
            features.append(float(gene.tpsl_gene.take_profit_pct or 0.1))
        else:
            features.append(0.05)
            features.append(0.1)

        # ポジションサイジングパラメータ
        if gene.position_sizing_gene and hasattr(
            gene.position_sizing_gene, "risk_per_trade"
        ):
            features.append(float(gene.position_sizing_gene.risk_per_trade or 0.01))
        else:
            features.append(0.01)

        # 2. 指標タイプベクトル（Bag of Words）
        # 利用可能な指標タイプの使用回数をカウント
        if self.indicator_types:
            indicator_vector = np.zeros(len(self.indicator_types))
            for ind in gene.indicators:
                if ind.type in self.indicator_map:
                    idx = self.indicator_map[ind.type]
                    indicator_vector[idx] += 1.0

            features.extend(indicator_vector.tolist())

        # 3. オペレータタイプベクトル（Bag of Words）
        if self.operator_types:
            operator_vector = np.zeros(len(self.operator_types))

            # 条件を収集（すべての条件リストから）
            all_conditions = []
            if gene.long_entry_conditions:
                all_conditions.extend(gene.long_entry_conditions)
            if gene.short_entry_conditions:
                all_conditions.extend(gene.short_entry_conditions)

            # 再帰的にカウント
            self._count_operators(all_conditions, operator_vector)

            features.extend(operator_vector.tolist())

        # 4. 時間軸特性（指標パラメータから推定）
        # period等のパラメータ値の平均と最大を取得し、戦略の時間的性質（短期/長期）を捉える
        period_values = []
        period_keys = [
            "period",
            "fast_period",
            "slow_period",
            "signal_period",
            "timeperiod",
            "k_period",
            "d_period",
        ]

        for ind in gene.indicators:
            for key in period_keys:
                if key in ind.parameters and isinstance(
                    ind.parameters[key], (int, float)
                ):
                    period_values.append(float(ind.parameters[key]))

        if period_values:
            features.append(float(np.mean(period_values)))
            features.append(float(np.max(period_values)))
        else:
            features.append(0.0)
            features.append(0.0)

        # 5. オペランド特性（定数比較 vs 動的比較）
        # 右辺が数値（定数）か、指標/価格（動的値）かの比率
        numeric_operands = 0.0
        dynamic_operands = 0.0

        all_conditions = []
        if gene.long_entry_conditions:
            all_conditions.extend(gene.long_entry_conditions)
        if gene.short_entry_conditions:
            all_conditions.extend(gene.short_entry_conditions)

        numeric_operands, dynamic_operands = self._count_operand_types(all_conditions)

        features.append(numeric_operands)
        features.append(dynamic_operands)

        return np.array(features)

    def _count_operators(self, conditions: List[Any], vector: np.ndarray):
        """条件リスト内のオペレータを再帰的にカウント"""
        for cond in conditions:
            if isinstance(cond, ConditionGroup):
                # ConditionGroupのoperator (AND/OR) もカウント
                if cond.operator and cond.operator in self.operator_map:
                    idx = self.operator_map[cond.operator]
                    vector[idx] += 1.0

                # ここでは内部のconditionsを再帰的に処理
                if cond.conditions:
                    self._count_operators(cond.conditions, vector)
            elif hasattr(cond, "operator"):
                # 通常のCondition
                op = cond.operator
                if op in self.operator_map:
                    idx = self.operator_map[op]
                    vector[idx] += 1.0

    def _count_operand_types(self, conditions: List[Any]) -> tuple[float, float]:
        """
        オペランドのタイプ（数値/動的）をカウント

        Args:
            conditions: 条件リスト

        Returns:
            (numeric_count, dynamic_count)
        """
        numeric = 0.0
        dynamic = 0.0

        for cond in conditions:
            if isinstance(cond, ConditionGroup):
                if cond.conditions:
                    n, d = self._count_operand_types(cond.conditions)
                    numeric += n
                    dynamic += d
            elif hasattr(cond, "right_operand"):
                # right_operand をチェック
                op_val = cond.right_operand

                # 数値型判定
                is_numeric = False
                if isinstance(op_val, (int, float)):
                    is_numeric = True
                elif isinstance(op_val, str):
                    try:
                        float(op_val)
                        is_numeric = True
                    except ValueError:
                        is_numeric = False

                if is_numeric:
                    numeric += 1.0
                else:
                    dynamic += 1.0

        return numeric, dynamic

    def _sharing_function(self, similarity: float) -> float:
        """
        共有関数

        Args:
            similarity: 類似度

        Returns:
            共有値
        """
        if similarity >= 0.0 and similarity <= self.sharing_radius:
            return 1.0  # 半径内では完全共有
        else:
            return 0.0  # 半径外では共有なし





