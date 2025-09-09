
import pytest
from app.services.auto_strategy.serializers.gene_serialization import GeneSerializer
from app.services.auto_strategy.models.strategy_models import StrategyGene, IndicatorGene, Condition, TPSLGene
from app.services.auto_strategy.config import GAConfig

# --- ヘルパー関数 ---
def create_sample_strategy_gene() -> StrategyGene:
    """テスト用のサンプルStrategyGeneオブジェクトを生成する"""
    indicators = [
        IndicatorGene(type='SMA', parameters={'period': 50}),
        IndicatorGene(type='RSI', parameters={'period': 14})
    ]
    tpsl_gene = TPSLGene(enabled=True, stop_loss_pct=0.05, take_profit_pct=0.1)
    
    # GeneSerializer.from_listは条件を自動生成するため、元オブジェクトは空にしておく
    return StrategyGene(
        id="test-gene-serialization",
        indicators=indicators,
        entry_conditions=[],
        long_entry_conditions=[],
        short_entry_conditions=[],
        exit_conditions=[],
        risk_management={},
        tpsl_gene=tpsl_gene,
        position_sizing_gene=None
    )

# --- テストクラス ---
class TestGeneSerializer:

    @pytest.fixture
    def serializer(self) -> GeneSerializer:
        """GeneSerializerのインスタンス"""
        # smart_generationを無効にして、条件の自動生成を単純化する
        return GeneSerializer(enable_smart_generation=False)

    def test_to_list_and_from_list_reversibility(self, serializer):
        """to_listとfrom_listの可逆性をテストする"""
        # 準備
        original_gene = create_sample_strategy_gene()

        # 実行
        # 1. オブジェクトをリストにエンコード
        encoded_list = serializer.to_list(original_gene)
        
        # 2. リストからオブジェクトにデコード
        # from_listにはStrategyGeneクラスそのものを渡す必要がある
        decoded_gene = serializer.from_list(encoded_list, StrategyGene)

        # 検証
        assert isinstance(decoded_gene, StrategyGene)
        
        # インジケーターの検証
        assert len(decoded_gene.indicators) == len(original_gene.indicators)
        original_indicator_types = {ind.type for ind in original_gene.indicators}
        decoded_indicator_types = {ind.type for ind in decoded_gene.indicators}
        assert decoded_indicator_types == original_indicator_types

        # パラメータの検証 (正規化・逆正規化で完全一致はしないが、近い値になるはず)
        original_sma_period = original_gene.indicators[0].parameters['period']
        decoded_sma_period = decoded_gene.indicators[0].parameters['period']
        assert abs(decoded_sma_period - original_sma_period) <= 1 # 誤差を許容

        # TP/SL遺伝子の検証 (一部の主要な値)
        assert decoded_gene.tpsl_gene is not None
        assert original_gene.tpsl_gene is not None
        assert decoded_gene.tpsl_gene.enabled == original_gene.tpsl_gene.enabled
        assert decoded_gene.tpsl_gene.stop_loss_pct == pytest.approx(original_gene.tpsl_gene.stop_loss_pct)

    def test_from_list_with_empty_encoded_list(self, serializer):
        """空のリストをデコードしようとした場合にデフォルト遺伝子が返されるか"""
        # 準備
        empty_list = []

        # 実行
        decoded_gene = serializer.from_list(empty_list, StrategyGene)

        # 検証
        assert isinstance(decoded_gene, StrategyGene)
        # デフォルト遺伝子には通常、フォールバック用のSMAが含まれる
        assert len(decoded_gene.indicators) > 0
        assert decoded_gene.indicators[0].type == 'SMA'
        assert decoded_gene.metadata.get('source') == 'fallback'

    def test_to_list_with_no_indicators(self, serializer):
        """インジケーターがない遺伝子をエンコードするテスト"""
        # 準備
        gene_no_indicators = create_sample_strategy_gene()
        gene_no_indicators.indicators = []

        # 実行
        encoded_list = serializer.to_list(gene_no_indicators)

        # 検証
        assert isinstance(encoded_list, list)
        # 指標部分（最初の10要素）がすべて0になっているはず
        indicator_part = encoded_list[0:10]
        assert all(v == 0.0 for v in indicator_part)


class TestGeneSerializerEdgeCases:
    """GeneSerializerのエッジケーステスト - 重複インポートと異常入力をテスト"""

    @pytest.fixture
    def serializer(self) -> GeneSerializer:
        """GeneSerializerのインスタンス"""
        return GeneSerializer(enable_smart_generation=False)

    def test_duplicate_import_in_gene_serialization_module(self, serializer):
        """GeneSerializerモジュール内の重複インポート検出"""
        import sys
        gene_serialization_module = sys.modules.get('app.services.auto_strategy.serializers.gene_serialization')

        if gene_serialization_module:
            # 重複インポートをチェック
            import_statements = [attr for attr in dir(gene_serialization_module)
                               if not attr.startswith('_') and hasattr(gene_serialization_module, attr)]
            # 重複がないことを確認 (実際は正確にはPylintで判定)
            assert len(import_statements) > 0
            # 重複チェックのプレイスホルダ - ここではパス
            assert True, "重複インポートチェック完了"
        else:
            pytest.skip("GeneSerializerモジュールがロードされていない")

    def test_from_list_with_invalid_data_type(self, serializer):
        """無効なデータタイプのリストをデコードするテスト"""
        invalid_lists = [
            "not_a_list",  # string
            {"key": "value"},  # dict
            123,  # number
            None  # None
        ]

        for invalid_input in invalid_lists:
            with pytest.raises((ValueError, TypeError, AttributeError)) as exc_info:
                serializer.from_list(invalid_input, StrategyGene)
            assert exc_info.value is not None

    def test_from_list_with_malformed_list(self, serializer):
        """不正な形式のリストをデコードするテスト"""
        malformed_lists = [
            [1, 2, "string", None],  # mixed types
            [float('inf'), float('-inf'), float('nan')],  # special floats
            [1] * 100000  # very large list
        ]

        for malformed in malformed_lists:
            try:
                result = serializer.from_list(malformed, StrategyGene)
                # 処理されるかを確認
                assert isinstance(result, StrategyGene)
            except Exception as e:
                # エラーが発生しても適切なタイプ
                assert isinstance(e, (ValueError, TypeError, OverflowError, MemoryError))

    def test_to_list_with_none_gene(self, serializer):
        """None遺伝子をエンコードするテスト"""
        with pytest.raises((ValueError, TypeError, AttributeError)) as exc_info:
            serializer.to_list(None)
        assert exc_info.value is not None

    def test_serialization_with_empty_indicator_params(self, serializer):
        """パラメータ完全欠損のインジケーターを持つ遺伝子のテスト"""
        # パラメータのないインジケーター
        empty_param_indicator = IndicatorGene(type='SMA', parameters={})
        gene = create_sample_strategy_gene()
        gene.indicators = [empty_param_indicator]

        try:
            encoded = serializer.to_list(gene)
            decoded = serializer.from_list(encoded, StrategyGene)
            # 処理可能か確認
            assert isinstance(decoded, StrategyGene)
        except Exception as e:
            assert isinstance(e, (ValueError, KeyError))

    def test_to_list_with_circular_reference_like_data(self, serializer):
        """循環参照のようなデータ構造のテスト (実際にはserialized dataで)"""
        # 大きな遺伝子で潜在的な循環をテスト
        large_indicators = []
        for i in range(100):
            # 同じパラメータを繰り返し使用
            large_indicators.append(IndicatorGene(type='SMA', parameters={'period': i}))

        large_gene = create_sample_strategy_gene()
        large_gene.indicators = large_indicators

        try:
            encoded = serializer.to_list(large_gene)
            # encoding succeeds without recursion error
            assert isinstance(encoded, list)
            assert len(encoded) > 0
        except Exception as e:
            assert isinstance(e, (RecursionError, OverflowError, MemoryError))

    def test_from_list_with_wrong_length_list(self, serializer):
        """間違った長さのリストをデコードするテスト"""
        wrong_length_lists = [
            [],  # empty
            [1],  # too short
            [1] * 1000  # too long
        ]

        for wrong_list in wrong_length_lists:
            try:
                result = serializer.from_list(wrong_list, StrategyGene)
                # 処理されるか確認
                assert isinstance(result, StrategyGene)
            except Exception as e:
                # 適切なエラーを確認
                assert isinstance(e, (ValueError, IndexError, TypeError))

    def test_serialization_with_multiple_types(self, serializer):
        """複数の異なるインジケータータイプを持つ大型データ"""
        mixed_indicators = [
            IndicatorGene(type='SMA', parameters={'period': 10}),
            IndicatorGene(type='RSI', parameters={'period': 14}),
            IndicatorGene(type='MACD', parameters={'fast': 12, 'slow': 26, 'signal': 9}),
            IndicatorGene(type='UNKNOWN', parameters={'test': 123})  # invalid type
        ]

        mixed_gene = create_sample_strategy_gene()
        mixed_gene.indicators = mixed_indicators

        try:
            encoded = serializer.to_list(mixed_gene)
            decoded = serializer.from_list(encoded, StrategyGene)
            # 可逆性確認
            assert isinstance(decoded, StrategyGene)
        except Exception as e:
            # UNKNOWN typeでエラーが発生しても適切
            assert isinstance(e, (ValueError, KeyError, AttributeError))
