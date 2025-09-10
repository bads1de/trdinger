"""
indicator_gene.py のユニットテスト
"""
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import replace

from backend.app.services.auto_strategy.models.indicator_gene import (
    IndicatorGene,
    IndicatorParams
)
from backend.app.services.auto_strategy.models.validator import GeneValidator
from backend.app.services.indicators.config.indicator_config import indicator_registry


class TestIndicatorGene:
    """IndicatorGene クラステスト"""

    def test_indicator_gene_initialization(self):
        """IndicatorGene の初期化テスト"""
        # 基本的な初期化
        gene = IndicatorGene(
            type="SMA",
            parameters={"period": 20},
            enabled=True
        )

        assert gene.type == "SMA"
        assert gene.parameters == {"period": 20}
        assert gene.enabled is True
        assert gene.json_config == {}
        assert isinstance(gene.parameters, dict)
        assert isinstance(gene.json_config, dict)

    def test_indicator_gene_default_values(self):
        """デフォルト値のテスト"""
        gene = IndicatorGene(type="RSI")

        assert gene.type == "RSI"
        assert gene.parameters == {}  # default_factory
        assert gene.enabled is True  # デフォルトTrue
        assert gene.json_config == {}  # default_factory

    def test_indicator_gene_validate_success(self):
        """validate() メソッドの成功ケーステスト"""
        gene = IndicatorGene(
            type="SMA",
            parameters={"period": 20},
            enabled=True
        )

        # IndicatorGene.validate() は GeneValidator を使用
        # GeneValidator をモックしてテスト
        with patch('backend.app.services.auto_strategy.models.indicator_gene.GeneValidator') as mock_validator:
            mock_validator_instance = MagicMock()
            mock_validator_instance.validate_indicator_gene.return_value = True
            mock_validator.return_value = mock_validator_instance

            result = gene.validate()
            assert result is True

            # GeneValidator が正しく呼び出されたか確認
            mock_validator.assert_called_once()
            mock_validator_instance.validate_indicator_gene.assert_called_once_with(gene)

    def test_indicator_gene_validate_failure(self):
        """validate() メソッドの失敗ケーステスト"""
        gene = IndicatorGene(
            type="UNKNOWN_INDICATOR",
            parameters={"invalid_param": "value"},
            enabled=False
        )

        with patch('backend.app.services.auto_strategy.models.indicator_gene.GeneValidator') as mock_validator:
            mock_validator_instance = MagicMock()
            mock_validator_instance.validate_indicator_gene.return_value = False
            mock_validator.return_value = mock_validator_instance

            result = gene.validate()
            assert result is False

            mock_validator_instance.validate_indicator_gene.assert_called_once_with(gene)

    def test_indicator_gene_validate_exception(self):
        """validate() メソッドでの例外処理テスト"""
        gene = IndicatorGene(type="SMA")

        with patch('backend.app.services.auto_strategy.models.validator.GeneValidator') as mock_validator:
            # GeneValidator のインスタンス化で例外
            mock_validator.side_effect = Exception("Test exception")

            result = gene.validate()
            assert result is False

    def test_indicator_gene_get_json_config_none_config(self):
        """get_json_config() メソッド: indicator_registry が None を返す場合"""
        gene = IndicatorGene(
            type="SMA",
            parameters={"period": 20}
        )

        with patch('app.services.indicators.config.indicator_registry') as mock_registry:
            # get_indicator_config が None を返す
            mock_registry.get_indicator_config.return_value = None

            result = gene.get_json_config()

            expected = {"indicator": "SMA", "parameters": {"period": 20}}
            assert result == expected
            mock_registry.get_indicator_config.assert_called_once_with("SMA")

    def test_indicator_gene_get_json_config_with_resolved_params(self):
        """get_json_config() メソッド: パラメータ解決テスト"""
        gene = IndicatorGene(
            type="SMA",
            parameters={"period": 30}  # デフォルト値20をオーバーライド
        )

        # Mock indicator config
        mock_config = MagicMock()
        mock_config.parameters = {
            "period": MagicMock(default_value=20),
            "source": MagicMock(default_value="close")
        }

        with patch('app.services.indicators.config.indicator_registry') as mock_registry:
            mock_registry.get_indicator_config.return_value = mock_config

            result = gene.get_json_config()

            expected = {
                "indicator": "SMA",
                "parameters": {
                    "period": 30,  # ユーザーの値を使用
                    "source": "close"  # デフォルト値を使用
                }
            }
            assert result == expected

    def test_indicator_gene_get_json_config_import_error(self):
        """get_json_config() メソッド: ImportError 時のフォールバックテスト"""
        gene = IndicatorGene(
            type="SMA",
            parameters={"period": 20}
        )

        # ImportError をシミュレート
        with patch('app.services.indicators.config.indicator_registry') as mock_registry:
            # インポート自体でエラーになる場合
            mock_registry.side_effect = ImportError("Module not found")

            result = gene.get_json_config()

            expected = {"indicator": "SMA", "parameters": {"period": 20}}
            assert result == expected

    def test_indicator_gene_get_json_config_partial_parameters(self):
        """get_json_config() メソッド: 部分的なパラメータテスト"""
        gene = IndicatorGene(
            type="RSI",
            parameters={"period": 14}  # 一部のパラメータのみ指定
        )

        mock_config = MagicMock()
        mock_config.parameters = {
            "period": MagicMock(default_value=14),
            "source": MagicMock(default_value="close"),
            "smoothing": MagicMock(default_value="sma")
        }

        with patch('app.services.indicators.config.indicator_registry') as mock_registry:
            mock_registry.get_indicator_config.return_value = mock_config

            result = gene.get_json_config()

            expected = {
                "indicator": "RSI",
                "parameters": {
                    "period": 14,  # ユーザーの値
                    "source": "close",  # デフォルト
                    "smoothing": "sma"  # デフォルト
                }
            }
            assert result == expected

    def test_indicator_gene_get_json_config_empty_parameters(self):
        """get_json_config() メソッド: 空のパラメータテスト"""
        gene = IndicatorGene(
            type="MACD",
            parameters={}  # パラメータなし
        )

        mock_config = MagicMock()
        mock_config.parameters = {
            "fast_period": MagicMock(default_value=12),
            "slow_period": MagicMock(default_value=26),
            "signal_period": MagicMock(default_value=9)
        }

        with patch('app.services.indicators.config.indicator_registry') as mock_registry:
            mock_registry.get_indicator_config.return_value = mock_config

            result = gene.get_json_config()

            expected = {
                "indicator": "MACD",
                "parameters": {
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_period": 9
                }
            }
            assert result == expected


class TestIndicatorParams:
    """IndicatorParams クラステスト"""

    def test_indicator_params_initialization(self):
        """IndicatorParams の初期化テスト"""
        params = IndicatorParams(
            indicator_type="SMA",
            period=20,
            source="close"
        )

        assert params.indicator_type == "SMA"
        assert params.period == 20
        assert params.source == "close"

    def test_indicator_params_default_values(self):
        """IndicatorParams のデフォルト値テスト"""
        # IndicatorParams にデフォルト値がないので最小パラメータでテスト
        with pytest.raises(TypeError):
            IndicatorParams()  # 必須パラメータなし

        # 正しい初期化
        params = IndicatorParams(indicator_type="RSI", period=14, source="close")
        assert params.indicator_type == "RSI"
        assert params.period == 14
        assert params.source == "close"


class TestIndicatorGeneEdgeCases:
    """エッジケーステスト"""

    def test_indicator_gene_with_none_type(self):
        """type が None の場合のテスト"""
        gene = IndicatorGene(type=None, parameters={})

        with patch('backend.app.services.auto_strategy.models.indicator_gene.GeneValidator') as mock_validator:
            mock_validator_instance = MagicMock()
            mock_validator_instance.validate_indicator_gene.return_value = False
            mock_validator.return_value = mock_validator_instance

            result = gene.validate()
            assert result is False

    def test_indicator_gene_with_non_dict_parameters(self):
        """parameters が辞書でない場合"""
        gene = IndicatorGene(type="SMA", parameters="invalid")  # 文字列

        # get_json_config の ImportError パスをテスト
        with patch('app.services.indicators.config.indicator_registry') as mock_registry:
            mock_registry.side_effect = ImportError()

            result = gene.get_json_config()
            # parameters が辞書でない場合はそのまま返す
            expected = {"indicator": "SMA", "parameters": "invalid"}
            assert result == expected

    def test_indicator_gene_config_fallback_chain(self):
        """設定取得のフォールバックチェーンをテスト"""
        gene = IndicatorGene(type="SMA", parameters={"period": 20})

        # 1. ImportError
        with patch('app.services.indicators.config.indicator_registry') as mock_registry:
            mock_registry.side_effect = ImportError("No module")

            result = gene.get_json_config()
            expected = {"indicator": "SMA", "parameters": {"period": 20}}
            assert result == expected

        # indicator_registry が利用可能だが get_indicator_config が None を返す場合
        with patch('app.services.indicators.config.indicator_registry') as mock_registry:
            mock_registry.get_indicator_config.return_value = None

            result = gene.get_json_config()
            assert result == expected

    def test_indicator_gene_empty_json_config(self):
        """json_config が空の状態でのテスト"""
        gene = IndicatorGene(type="SMA")
        assert gene.json_config == {}

        # json_config を変更しても json_config 属性自体は自動更新されない
        gene.json_config["test"] = "value"
        assert gene.json_config == {"test": "value"}

    def test_indicator_gene_enabled_flag(self):
        """enabled フラグのテスト"""
        enabled_gene = IndicatorGene(type="RSI", enabled=True)
        disabled_gene = IndicatorGene(type="RSI", enabled=False)

        assert enabled_gene.enabled is True
        assert disabled_gene.enabled is False

        # デフォルトでは True
        default_gene = IndicatorGene(type="RSI")
        assert default_gene.enabled is True