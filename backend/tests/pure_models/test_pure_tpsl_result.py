"""
Test for TPSLResult
"""
import pytest
from backend.app.services.auto_strategy.models.tpsl_result import TPSLResult


class TestTPSLResult:
    def test_init_with_required_params(self):
        result = TPSLResult(
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
            method_used="test_method"
        )

        assert result.stop_loss_pct == 0.02
        assert result.take_profit_pct == 0.05
        assert result.method_used == "test_method"
        assert result.confidence_score == 0.0  # default

    def test_post_init_defaults(self):
        result = TPSLResult(
            stop_loss_pct=0.01,
            take_profit_pct=0.03,
            method_used="method",
            confidence_score=0.8
        )

        assert result.expected_performance == {}
        assert result.metadata == {}

    def test_to_dict(self):
        result = TPSLResult(
            stop_loss_pct=0.025,
            take_profit_pct=0.06,
            method_used="risk_reward",
            confidence_score=0.7,
            expected_performance={"win_rate": 0.6},
            metadata={"source": "test"}
        )

        dict_result = result.to_dict()

        assert dict_result["stop_loss_pct"] == 0.025
        assert dict_result["take_profit_pct"] == 0.06
        assert dict_result["method_used"] == "risk_reward"
        assert dict_result["confidence_score"] == 0.7
        assert dict_result["expected_performance"] == {"win_rate": 0.6}
        assert dict_result["metadata"] == {"source": "test"}

    def test_to_dict_with_none_values(self):
        # expected_performance と metadata が None の場合
        result = TPSLResult(
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            method_used="fixed"
        )

        dict_result = result.to_dict()

        # __post_init__ で None が {} に変換されることを確認
        assert dict_result["expected_performance"] == {}
        assert dict_result["metadata"] == {}

    def test_init_with_none_values_explicit(self):
        # 明示的に None を渡す場合
        result = TPSLResult(
            stop_loss_pct=0.03,
            take_profit_pct=0.07,
            method_used="volatility",
            expected_performance=None,
            metadata=None
        )

        # 初期化後は空辞書になっているはず
        assert result.expected_performance == {}
        assert result.metadata == {}

    def test_init_with_complex_metadata(self):
        # 複雑なメタデータを含むテスト
        metadata = {
            "timestamp": "2023-01-01T00:00:00",
            "calculation_method": "atr_based",
            "parameters": {
                "multiplier": 2.0,
                "period": 14
            },
            "warnings": ["extreme_values"],
            "version": "1.0.0"
        }

        expected_performance = {
            "expected_return": 0.05,
            "max_drawdown": 0.02,
            "sharpe_ratio": 1.5,
            "win_rate": 0.65,
            "profit_factor": 1.8
        }

        result = TPSLResult(
            stop_loss_pct=0.04,
            take_profit_pct=0.08,
            method_used="statistical",
            confidence_score=0.85,
            expected_performance=expected_performance,
            metadata=metadata
        )

        assert result.stop_loss_pct == 0.04
        assert result.take_profit_pct == 0.08
        assert result.method_used == "statistical"
        assert result.confidence_score == 0.85
        assert result.expected_performance == expected_performance
        assert result.metadata == metadata

    def test_to_dict_with_complex_data(self):
        # 複雑なto_dictテスト
        metadata = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "bool": True,
            "float": 1.5
        }

        expected_performance = {
            "nested_metric": {"sub_metric": 0.8},
            "array_values": [0.1, 0.2, 0.3]
        }

        result = TPSLResult(
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            method_used="adaptive",
            confidence_score=0.9,
            expected_performance=expected_performance,
            metadata=metadata
        )

        dict_result = result.to_dict()

        # ネストされた構造が正しく保存されていることを確認
        assert dict_result["expected_performance"]["nested_metric"]["sub_metric"] == 0.8
        assert dict_result["metadata"]["nested"]["key"] == "value"
        assert dict_result["metadata"]["list"] == [1, 2, 3]
        assert dict_result["metadata"]["bool"] == True
        assert dict_result["metadata"]["float"] == 1.5

    def test_boundary_values(self):
        # 境界値テスト
        result_min = TPSLResult(
            stop_loss_pct=0.0,
            take_profit_pct=0.001,
            method_used="test",
            confidence_score=-1.0
        )

        result_max = TPSLResult(
            stop_loss_pct=1.0,
            take_profit_pct=2.0,
            method_used="test",
            confidence_score=2.0
        )

        assert result_min.stop_loss_pct == 0.0
        assert result_max.stop_loss_pct == 1.0

    def test_empty_strings(self):
        # 空文字列のテスト
        result = TPSLResult(
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            method_used=""  # 空文字列
        )

        assert result.method_used == ""

    def test_post_init_behavior(self):
        # __post_init__ の独自動作テスト
        result = TPSLResult(
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            method_used="test_method",
            confidence_score=0.5
        )

        # expected_performance と metadata をチェック
        assert result.expected_performance == {}
        assert result.metadata == {}

        # 明示的に値をセットした後
        result.expected_performance["test_key"] = "test_value"
        result.metadata["meta_key"] = 42

        assert result.expected_performance["test_key"] == "test_value"
        assert result.metadata["meta_key"] == 42

    def test_dataclass_features(self):
        # dataclassとしての基本機能テスト
        result1 = TPSLResult(
            stop_loss_pct=0.04,
            take_profit_pct=0.08,
            method_used="fixed",
            confidence_score=0.5
        )

        result2 = TPSLResult(
            stop_loss_pct=0.04,
            take_profit_pct=0.08,
            method_used="fixed",
            confidence_score=0.75  # 異なる値
        )

        # dataclassで自動生成される __eq__ のテスト
        assert result1 == TPSLResult(
            stop_loss_pct=0.04,
            take_profit_pct=0.08,
            method_used="fixed",
            confidence_score=0.5  # 同じ値
        )

        # dataclassで自動生成される __repr__ のテスト
        repr_str = repr(result1)
        assert "stop_loss_pct=0.04" in repr_str
        assert "take_profit_pct=0.08" in repr_str
        assert "method_used='fixed'" in repr_str
        assert "confidence_score=0.5" in repr_str

        # 不等テスト - confidence_score が異なる
        assert result1 != result2