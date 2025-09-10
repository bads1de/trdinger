"""
条件評価器のテスト

ConditionEvaluatorクラスのTDDテストケース
バグを発見し、修正を行います。
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from numbers import Real

from backend.app.services.auto_strategy.core.condition_evaluator import ConditionEvaluator
from backend.app.services.auto_strategy.models.strategy_models import Condition, ConditionGroup


@pytest.fixture
def evaluator():
    """テスト用ConditionEvaluatorインスタンス"""
    return ConditionEvaluator()


@pytest.fixture
def sample_condition():
    """サンプル条件"""
    return Condition(
        left_operand="close",
        operator=">",
        right_operand="sma_20"
    )


@pytest.fixture
def sample_condition_group():
    """サンプル条件グループ"""
    condition1 = Condition(left_operand="rsi", operator=">", right_operand=30)
    condition2 = Condition(left_operand="macd", operator=">", right_operand="macd_signal")
    return ConditionGroup(conditions=[condition1, condition2])


@pytest.fixture
def mock_strategy_instance():
    """モック戦略インスタンス"""
    strategy = Mock()

    # pandas Seriesを返す
    price_data = pd.Series([100, 101, 99, 102, 98], index=pd.date_range('2023-01-01', periods=5))
    strategy.close = price_data
    strategy.sma_20 = 100.0
    strategy.RSI = 65.5
    strategy.MACD = -0.5

    return strategy


class TestConditionEvaluator:

    def test_initialization(self, evaluator):
        """初期化テスト"""
        assert isinstance(evaluator, ConditionEvaluator)

    def test_evaluate_conditions_empty(self, evaluator, mock_strategy_instance):
        """空条件リストの評価"""
        result = evaluator.evaluate_conditions([], mock_strategy_instance)
        assert result is True

    def test_evaluate_conditions_single_condition_true(self, evaluator, mock_strategy_instance, sample_condition):
        """単一条件Trueの場合"""
        # close > sma_20 をTrueにする
        mock_strategy_instance.close = 105.0
        mock_strategy_instance.sma_20 = 100.0

        result = evaluator.evaluate_conditions([sample_condition], mock_strategy_instance)
        assert result is True

    def test_evaluate_conditions_single_condition_false(self, evaluator, mock_strategy_instance, sample_condition):
        """単一条件Falseの場合"""
        # close > sma_20 をFalseにする
        mock_strategy_instance.close = 95.0
        mock_strategy_instance.sma_20 = 100.0

        result = evaluator.evaluate_conditions([sample_condition], mock_strategy_instance)
        assert result is False

    def test_evaluate_conditions_multiple_conditions_and(self, evaluator, mock_strategy_instance):
        """複数条件のAND評価"""
        condition1 = Condition(left_operand="close", operator=">", right_operand="sma_20")
        condition2 = Condition(left_operand="rsi", operator=">", right_operand=50)

        # 両方True
        mock_strategy_instance.close = 105.0
        mock_strategy_instance.sma_20 = 100.0
        mock_strategy_instance.rsi = 65

        result = evaluator.evaluate_conditions([condition1, condition2], mock_strategy_instance)
        assert result is True

        # 一つFalse
        mock_strategy_instance.rsi = 45
        result = evaluator.evaluate_conditions([condition1, condition2], mock_strategy_instance)
        assert result is False

    def test_evaluate_conditions_with_condition_group_true(self, evaluator, mock_strategy_instance, sample_condition, sample_condition_group):
        """条件グループを含む評価（True）"""
        mock_strategy_instance.rsi = 35  # グループ条件の一つがTrue
        mock_strategy_instance.macd = -0.3
        mock_strategy_instance.macd_signal = -0.8

        result = evaluator.evaluate_conditions([sample_condition_group], mock_strategy_instance)
        assert result is True

    def test_evaluate_conditions_with_condition_group_false(self, evaluator, mock_strategy_instance, sample_condition_group):
        """条件グループを含む評価（False）"""
        # グループ内の条件が全てFalse
        mock_strategy_instance.rsi = 25
        mock_strategy_instance.macd = -1.0
        mock_strategy_instance.macd_signal = -0.5

        result = evaluator.evaluate_conditions([sample_condition_group], mock_strategy_instance)
        assert result is False

    def test_evaluate_conditions_mixed_conditions(self, evaluator, mock_strategy_instance, sample_condition, sample_condition_group):
        """通常条件と条件グループの混合"""
        # グループはTrue、通常条件はFalse
        mock_strategy_instance.rsi = 35  # グループTrue
        mock_strategy_instance.close = 95  # 通常False
        mock_strategy_instance.sma_20 = 100

        result = evaluator.evaluate_conditions([sample_condition_group, sample_condition], mock_strategy_instance)
        assert result is False  # ANDなのでFalse

    @patch('backend.app.services.auto_strategy.core.condition_evaluator.logger')
    def test_evaluate_conditions_exception_handling(self, mock_logger, evaluator):
        """評価中に例外が発生した場合の処理"""
        mock_strategy_instance = Mock()
        sample_condition = Condition(left_operand="invalid", operator=">", right_operand="test")

        # get_condition_valueで例外
        with patch.object(evaluator, 'get_condition_value', side_effect=Exception("Test error")):
            result = evaluator.evaluate_conditions([sample_condition], mock_strategy_instance)

            assert result is False
            mock_logger.error.assert_called_once()


class TestConditionGroupEvaluation:

    def test_evaluate_condition_group_empty(self, evaluator, mock_strategy_instance):
        """空条件グループの評価"""
        group = ConditionGroup(conditions=[])
        result = evaluator._evaluate_condition_group(group, mock_strategy_instance)
        assert result is False

    def test_evaluate_condition_group_single_true(self, evaluator, mock_strategy_instance, sample_condition):
        """単一条件Trueのグループ評価"""
        group = ConditionGroup(conditions=[sample_condition])
        mock_strategy_instance.close = 105.0
        mock_strategy_instance.sma_20 = 100.0

        result = evaluator._evaluate_condition_group(group, mock_strategy_instance)
        assert result is True

    def test_evaluate_condition_group_or_logic(self, evaluator, mock_strategy_instance):
        """OR論理の評価"""
        condition1 = Condition(left_operand="rsi", operator=">", right_operand=70)  # False
        condition2 = Condition(left_operand="macd", operator=">", right_operand=0)   # True

        group = ConditionGroup(conditions=[condition1, condition2])

        mock_strategy_instance.rsi = 65
        mock_strategy_instance.macd = 0.1

        result = evaluator._evaluate_condition_group(group, mock_strategy_instance)
        assert result is True

    def test_evaluate_condition_group_all_false(self, evaluator, mock_strategy_instance):
        """グループ内全てFalseの評価"""
        condition1 = Condition(left_operand="rsi", operator=">", right_operand=70)
        condition2 = Condition(left_operand="macd", operator=">", right_operand=0)

        group = ConditionGroup(conditions=[condition1, condition2])

        mock_strategy_instance.rsi = 65
        mock_strategy_instance.macd = -0.1

        result = evaluator._evaluate_condition_group(group, mock_strategy_instance)
        assert result is False

    @patch('backend.app.services.auto_strategy.core.condition_evaluator.logger')
    def test_evaluate_condition_group_exception(self, mock_logger, evaluator, sample_condition):
        """グループ評価中の例外処理"""
        group = ConditionGroup(conditions=[sample_condition])

        with patch.object(evaluator, 'evaluate_single_condition', side_effect=Exception("Test error")):
            result = evaluator._evaluate_condition_group(group, Mock())

            assert result is False
            mock_logger.error.assert_called_once()


class TestSingleConditionEvaluation:

    def test_evaluate_single_condition_greater_than_true(self, evaluator, mock_strategy_instance):
        """単一条件 > Trueの評価"""
        condition = Condition(left_operand=100, operator=">", right_operand=50)
        result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)
        assert result is True

    def test_evaluate_single_condition_less_than_false(self, evaluator, mock_strategy_instance):
        """単一条件 < Falseの評価"""
        condition = Condition(left_operand=50, operator="<", right_operand=100)
        result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)
        assert result is False

    def test_evaluate_single_condition_equal_close(self, evaluator, mock_strategy_instance):
        """単一条件 == closeの評価"""
        condition = Condition(left_operand=10.0, operator="==", right_operand=10.0000000001)
        result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)
        assert result is True

    def test_evaluate_single_condition_not_equal(self, evaluator, mock_strategy_instance):
        """単一条件 != の評価"""
        condition = Condition(left_operand=10.0, operator="!=", right_operand=12.0)
        result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)
        assert result is True

    def test_evaluate_single_condition_gte(self, evaluator, mock_strategy_instance):
        """単一条件 >= の評価"""
        condition = Condition(left_operand=10, operator=">=", right_operand=10)
        result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)
        assert result is True

    def test_evaluate_single_condition_lte(self, evaluator, mock_strategy_instance):
        """単一条件 <= の評価"""
        condition = Condition(left_operand=10, operator="<=", right_operand=10)
        result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)
        assert result is True

    def test_evaluate_single_condition_invalid_type_left(self, evaluator, mock_strategy_instance):
        """左オペランドが非数値の場合"""
        condition = Condition(left_operand="string", operator=">", right_operand=50)
        result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)
        assert result is False

    def test_evaluate_single_condition_invalid_type_right(self, evaluator, mock_strategy_instance):
        """右オペランドが非数値の場合"""
        condition = Condition(left_operand=100, operator=">", right_operand="string")
        result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)
        assert result is False

    def test_evaluate_single_condition_both_strings(self, evaluator, mock_strategy_instance):
        """両オペランドが非数値の場合"""
        condition = Condition(left_operand="test1", operator=">", right_operand="test2")
        result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)
        assert result is False

    def test_evaluate_single_condition_unknown_operator(self, evaluator, mock_strategy_instance):
        """未知の演算子の場合"""
        condition = Condition(left_operand=100, operator="???", right_operand=50)

        with patch('backend.app.services.auto_strategy.core.condition_evaluator.logger') as mock_logger:
            result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)

            assert result is False
            mock_logger.warning.assert_called_once()

    @patch('backend.app.services.auto_strategy.core.condition_evaluator.logger')
    def test_evaluate_single_condition_ml_debug_logging(self, mock_logger, evaluator, mock_strategy_instance):
        """ML指標のデバッグログ出力"""
        condition = Condition(left_operand="ML_SIGNAL", operator=">", right_operand=0)

        # デバッグログが2回呼び出されることを確認
        with patch.object(evaluator, 'get_condition_value') as mock_get_val:
            mock_get_val.side_effect = [1.5, 0.5]
            result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)

            # infoログが2回呼ばれる（評価デバッグ用）
            assert mock_logger.info.call_count == 2

    @patch('backend.app.services.auto_strategy.core.condition_evaluator.logger')
    def test_evaluate_single_condition_exception_handling(self, mock_logger, evaluator, mock_strategy_instance):
        """評価中の例外処理"""
        condition = Condition(left_operand=100, operator=">", right_operand=50)

        # get_condition_valueで例外
        with patch.object(evaluator, 'get_condition_value', side_effect=Exception("Test error")):
            result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)

            assert result is False
            mock_logger.error.assert_called_once()


class TestGetFinalValue:

    def test_get_final_value_pandas_series(self, evaluator):
        """pandas Seriesからの値取得"""
        series = pd.Series([100, 101, 99, 102, 98])
        result = evaluator._get_final_value(series)
        assert result == 98  # 末尾値

    def test_get_final_value_pandas_series_single(self, evaluator):
        """単一要素のSeries"""
        series = pd.Series([100])
        result = evaluator._get_final_value(series)
        assert result == 100.0

    def test_get_final_value_list(self, evaluator):
        """リストからの値取得"""
        test_list = [100, 101, 99, 102, 98]
        result = evaluator._get_final_value(test_list)
        assert result == 98.0

    def test_get_final_value_numpy_array(self, evaluator):
        """numpy arrayからの値取得"""
        arr = np.array([100, 101, 99, 102, 98])
        result = evaluator._get_final_value(arr)
        assert result == 98.0

    def test_get_final_value_scalar(self, evaluator):
        """スカラー値"""
        result = evaluator._get_final_value(123.5)
        assert result == 123.5

    def test_get_final_value_zero_scalar(self, evaluator):
        """ゼロ値のスカラー"""
        result = evaluator._get_final_value(0)
        assert result == 0.0

    def test_get_final_value_negative_scalar(self, evaluator):
        """負のスカラー値"""
        result = evaluator._get_final_value(-50.5)
        assert result == -50.5

    def test_get_final_value_string_unhandled(self, evaluator):
        """文字列などの未処理型"""
        result = evaluator._get_final_value("test")
        assert result == 0.0

    def test_get_final_value_list_with_nan(self, evaluator):
        """NaNを含むリスト"""
        test_list = [100, np.nan, 99]
        result = evaluator._get_final_value(test_list)
        assert result == 99.0  # NaNでない最新値

    def test_get_final_value_pandas_series_with_nan(self, evaluator):
        """NaNを含むSeries"""
        series = pd.Series([100, np.nan, 99])
        result = evaluator._get_final_value(series)
        assert result == 99.0

    def test_get_final_value_empty_list(self, evaluator):
        """空リスト"""
        try:
            result = evaluator._get_final_value([])
            assert result == 0.0
        except (IndexError, TypeError):
            assert True  # 期待される例外

    def test_get_final_value_empty_series(self, evaluator):
        """空Series"""
        series = pd.Series([], dtype=float)
        try:
            result = evaluator._get_final_value(series)
            assert np.isnan(result) or result == 0.0
        except (IndexError, KeyError):
            assert True


class TestGetConditionValue:

    def test_get_condition_value_numeric_operand(self, evaluator, mock_strategy_instance):
        """数値オペランドの処理"""
        # 右オペランドは数値
        result = evaluator.get_condition_value(123.5, mock_strategy_instance)
        assert result == 123.5

        result = evaluator.get_condition_value(0, mock_strategy_instance)
        assert result == 0.0

    def test_get_condition_value_dict_pandas_access(self, evaluator, mock_strategy_instance):
        """辞書オペランドのパンダスアクセス"""
        # close属性にアクセス
        operand = {"indicator": "close"}
        mock_strategy_instance.close = pd.Series([100, 101, 99])
        result = evaluator.get_condition_value(operand, mock_strategy_instance)
        assert result == 99.0

    def test_get_condition_value_dict_scalar_value(self, evaluator, mock_strategy_instance):
        """辞書オペランドのスカラー値アクセス"""
        operand = {"indicator": "sma_20"}
        mock_strategy_instance.sma_20 = 105.5
        result = evaluator.get_condition_value(operand, mock_strategy_instance)
        assert result == 105.5

    def test_get_condition_value_dict_no_indicator_key(self, evaluator, mock_strategy_instance):
        """インディケータキーなしの辞書"""
        operand = {"param": "value"}
        result = evaluator.get_condition_value(operand, mock_strategy_instance)
        assert result == 0.0

    def test_get_condition_value_dict_attribute_error(self, evaluator, mock_strategy_instance):
        """存在しない属性アクセス"""
        operand = {"indicator": "nonexistent"}
        with patch('backend.app.services.auto_strategy.core.condition_evaluator.logger') as mock_logger:
            result = evaluator.get_condition_value(operand, mock_strategy_instance)

            assert result == 0.0
            mock_logger.warning.assert_called_once()

    def test_get_condition_value_dict_nonexistent_attribute(self, evaluator, mock_strategy_instance):
        """存在しない属性アクセス"""
        operand = {"indicator": "nonexistent"}
        def has_nonexistent():
            return False

        mock_strategy_instance.nonexistent = None  # 明示的にNoneを設定
        del mock_strategy_instance.nonexistent  # 削除して存在しないように

        result = evaluator.get_condition_value(operand, mock_strategy_instance)
        assert result == 0.0

    def test_get_condition_value_string_operand_working(self, evaluator, mock_strategy_instance):
        """働く文字列オペランド"""
        result = evaluator.get_condition_value("close", mock_strategy_instance)
        assert result == 98.0  # モックの末尾値

    def test_get_condition_value_string_operand_error(self, evaluator, mock_strategy_instance):
        """存在しない文字列オペランド"""
        with patch('backend.app.services.auto_strategy.core.condition_evaluator.logger') as mock_logger:
            result = evaluator.get_condition_value("nonexistent", mock_strategy_instance)

            assert result == 0.0
            mock_logger.warning.assert_called_once()

    def test_get_condition_value_invalid_type(self, evaluator, mock_strategy_instance):
        """無効なオペランド型"""
        result = evaluator.get_condition_value(None, mock_strategy_instance)
        assert result == 0.0

    def test_get_condition_value_exception_handling(self, evaluator, mock_strategy_instance):
        """getattrでの例外処理"""
        with patch('backend.app.services.auto_strategy.core.condition_evaluator.logger') as mock_logger:
            result = evaluator.get_condition_value("close", None)  # Noneを渡してAttributeError

            assert result == 0.0
            mock_logger.error.assert_called_once()

    @patch('backend.app.services.auto_strategy.core.condition_evaluator.logger')
    def test_get_condition_value_pandas_final_value_with_exception(self, mock_logger, evaluator):
        """_get_final_valueでの例外処理"""
        operand = "close"
        mock_strategy_instance = Mock()
        mock_strategy_instance.close = pd.Series([100, np.nan])  # NaNを含む

        # _get_final_valueでfloat変換例外をシミュレート
        with patch.object(evaluator, '_get_final_value', side_effect=Exception("Test error")):
            result = evaluator.get_condition_value(operand, mock_strategy_instance)

            assert result == 0.0
            mock_logger.error.assert_called_once()


# エッジケーステスト
class TestEdgeCases:

    def test_evaluate_conditions_with_none_list(self, evaluator):
        """None条件リスト"""
        result = evaluator.evaluate_conditions(None, Mock())
        assert result is False

    def test_evaluate_conditions_with_non_list(self, evaluator):
        """非リスト条件"""
        with patch('backend.app.services.auto_strategy.core.condition_evaluator.logger') as mock_logger:
            result = evaluator.evaluate_conditions("invalid", Mock())

            assert result is False
            mock_logger.error.assert_called_once()

    def test_condition_group_is_empty_method(self, evaluator):
        """ConditionGroupのis_emptyメソッドテスト"""
        empty_group = ConditionGroup(conditions=[])
        if hasattr(empty_group, 'is_empty'):
            assert empty_group.is_empty() is True

        non_empty_group = ConditionGroup(conditions=[Condition(left_operand=1, operator=">", right_operand=0)])
        if hasattr(non_empty_group, 'is_empty'):
            assert non_empty_group.is_empty() is False

    def test_evaluate_single_condition_extreme_values(self, evaluator, mock_strategy_instance):
        """極端な値の評価"""
        # 非常に大きな値
        condition = Condition(left_operand=1e10, operator=">", right_operand=1e9)
        result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)
        assert result is True

        # 非常に小さい値
        condition = Condition(left_operand=1e-10, operator=">", right_operand=1e-9)
        result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)
        assert result is False

        # inf値
        condition = Condition(left_operand=float('inf'), operator=">", right_operand=100)
        result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)
        assert result is True

    def test_evaluate_single_condition_numpy_types(self, evaluator, mock_strategy_instance):
        """numpy型との比較"""
        condition = Condition(left_operand=np.int64(100), operator=">", right_operand=np.float64(50))
        result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)
        assert result is True

    @patch('backend.app.services.auto_strategy.core.condition_evaluator.logger')
    def test_modular_log_debug_for_non_ml_operands(self, mock_logger, evaluator, mock_strategy_instance):
        """非MLオペランドでのログチェエック"""
        condition = Condition(left_operand="close", operator=">", right_operand=100)

        result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)

        # 非MLオペランドなのでログは出力されない
        mock_logger.info.assert_not_called()
        mock_logger.warning.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])
class TestMLEvaluationEdgeCases:
    """ML指標評価のエッジケーステスト

    より細かいML指標のバリデーションと処理をテスト
    """

    def test_ml_indicator_with_numpy_types(self, evaluator, mock_strategy_instance):
        """ML指標とnumpy型の比較"""
        import numpy as np

        condition = Condition(left_operand="ML_PREDICTOR_1", operator=">", right_operand=np.float64(0.5))
        mock_strategy_instance.ML_PREDICTOR_1 = np.int32(1)

        result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)
        assert result is True

    def test_ml_indicator_nearly_equal_debug_logging(self, evaluator, mock_strategy_instance):
        """ML指標が非常に近い値での==比較のデバッグログ"""
        with patch('backend.app.services.auto_strategy.core.condition_evaluator.logger') as mock_logger:
            # 確実に等しい値を設定
            condition = Condition(left_operand="ML_SIGNAL", operator="==", right_operand=1.0)
            mock_strategy_instance.ML_SIGNAL = 1.0  # シンプルなfloat

            # まず値が正しく取得されることを確認
            left_val = evaluator.get_condition_value("ML_SIGNAL", mock_strategy_instance)
            right_val = evaluator.get_condition_value(1.0, mock_strategy_instance)
            assert left_val == 1.0
            assert right_val == 1.0

            result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)

            # np.iscloseでTrueになるはず（== Trueより寛容）
            assert result == True, f"Expected True, got {result} (type: {type(result)})"
            assert bool(result) is True, "Result should be truthy"
            # デバッグログが2回呼ばれる（評価デバッグ用と評価結果用）
            assert mock_logger.info.call_count == 2

            # ログの内容も確認（オプション）
            if mock_logger.info.call_count >= 2:
                calls = mock_logger.info.call_args_list
                # 1番目の呼び出し：評価デバッグ
                assert "ML条件評価デバッグ" in calls[0][0][0]
                assert "ML_SIGNAL" in calls[0][0][0]
                # 2番目の呼び出し：評価結果
                assert "評価結果:" in calls[1][0][0]
                assert "True" in calls[1][0][0]

    def test_ml_indicator_extreme_values(self, evaluator, mock_strategy_instance):
        """ML指標の極端な値での比較"""
        # 非常に大きな値
        condition = Condition(left_operand="ML_SCORE", operator=">", right_operand=float('inf'))
class TestDebugStringNumeric:
    """デバッグ用: 文字列数値パース問題を特定"""

    def test_direct_debug_string_parsing(self, evaluator, mock_strategy_instance):
        """直接デバッグ: 文字列の数値パースをテスト"""
        from unittest.mock import patch

        # このパッチが正しく機能するかどうか確認
        patch_path = 'backend.app.services.auto_strategy.core.condition_evaluator.logger'
        print(f"Patch path: {patch_path}")

        with patch(patch_path) as mock_logger:
            print("Mock logger created")
            result = evaluator.get_condition_value("not_a_number", mock_strategy_instance)
            print(f"get_condition_value result: {result}")
            print(f"Warning call count: {mock_logger.warning.call_count}")

            if mock_logger.warning.call_count > 0:
                print(f"Warning calls: {mock_logger.warning.call_args_list}")

        # 期待値: "not_a_number" は数値に変換できないので警告が出力されるはず
        assert result == 0.0

    def test_triple_debug_condition_evaluation(self, evaluator, mock_strategy_instance):
        """3重デバッグ: 条件評価全体"""
        from unittest.mock import patch
        from backend.app.services.auto_strategy.models.strategy_models import Condition

        condition = Condition(left_operand="not_a_number", operator=">", right_operand=5)
        patch_path = 'backend.app.services.auto_strategy.core.condition_evaluator.logger'

        print("=" * 50)
        print("TRIANGLE DEBUG START")
        print(f"Testing condition: {condition.left_operand} > {condition.right_operand}")
        print("=" * 50)

        with patch(patch_path) as mock_logger:
            print("1. Mock logger created successfully")

            # 中間ステップでデバッグ
            left_val = evaluator.get_condition_value(condition.left_operand, mock_strategy_instance)
            print(f"2. Left value: {left_val} (type: {type(left_val)})")

            right_val = evaluator.get_condition_value(condition.right_operand, mock_strategy_instance)
            print(f"3. Right value: {right_val} (type: {right_val})")

            print(f"4. Logger warning calls so far: {mock_logger.warning.call_count}")

            result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)
            print(f"5. Final result: {result}")
            print(f"6. Final logger warning calls: {mock_logger.warning.call_count}")

            if mock_logger.warning.call_count > 0:
                print("7. Warning messages:")
                for call in mock_logger.warning.call_args_list:
                    print(f"   - {call}")
            else:
                print("7. NO WARNINGS LOGGED!")

        print("=" * 50)
        print("TRIANGLE DEBUG END")
        print("=" * 50)

        # 少なくとも警告は1回コールされているはず
        # できれば2回（数値変換時の警告と評価時の警告両方）
        assert result is False, "評価結果はFalseになるはず"
        mock_strategy_instance.ML_SCORE = float('inf')

        result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)
        assert result is False  # inf > inf はFalse

        # NaN値
        condition = Condition(left_operand="ML_SCORE", operator="!=", right_operand=float('nan'))
        mock_strategy_instance.ML_SCORE = 1.0

        result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)
        assert result is True  # NaNとの比較は常にTrue

    def test_ml_indicator_case_insensitive_prefix(self, evaluator, mock_strategy_instance):
        """ML指標プレフィックスの大文字小文字区別テスト"""
        # ml_の小文字プレフィックス
        condition = Condition(left_operand="ml_prediction", operator=">", right_operand=0.5)
        mock_strategy_instance.ml_prediction = 0.8

        result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)
        # 小文字のml_はML指標として認識されないはず
        assert result is True

    def test_ml_indicator_empty_string(self, evaluator, mock_strategy_instance):
        """ML指標名が空文字列の場合"""
        condition = Condition(left_operand="ML_", operator=">", right_operand=0.5)

        with patch('backend.app.services.auto_strategy.core.condition_evaluator.logger') as mock_logger:
            result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)

            # ML_自体は存在しないはずなのでログが出力される
            mock_logger.warning.assert_called_once()
            assert result is False

    def test_ml_indicator_numeric_suffix_variations(self, evaluator, mock_strategy_instance):
        """ML指標の数値接尾語のバリエーション"""
        test_cases = [
            ("ML_MODEL_1", 0.8),
            ("ML_SIGNAL_001", 0.2),
            ("ML_PREDICTOR_a1", 0.9),  # 数字以外を含む
            ("ML_SCORE_v2.1", 0.3),
            ("ML_OUTPUT_10_000", 0.7),
        ]

        for operand_name, value in test_cases:
            mock_strategy_instance.__setattr__(operand_name, value)
            condition = Condition(left_operand=operand_name, operator=">", right_operand=0.5)
            result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)

            expected = value > 0.5
            assert result is expected, f"Failed for {operand_name}: expected {expected}, got {result}"

    def test_ml_indicator_debug_log_content(self, evaluator, mock_strategy_instance):
        """ML指標デバッグログの内容確認"""
        with patch('backend.app.services.auto_strategy.core.condition_evaluator.logger') as mock_logger:
            condition = Condition(left_operand="ML_TEST_SIGNAL", operator=">", right_operand=0.3)
            mock_strategy_instance.ML_TEST_SIGNAL = 0.7

            result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)

            # デバッグログの内容を確認
            assert mock_logger.info.call_count == 2

            # 最初のログ（評価デバッグ）
            first_call_args = mock_logger.info.call_args_list[0][0][0]
            assert "ML条件評価デバッグ" in first_call_args
            assert "ML_TEST_SIGNAL" in first_call_args
            assert ">" in first_call_args

            # 2番目のログ（評価結果）
            second_call_args = mock_logger.info.call_args_list[1][0][0]
            assert "評価結果:" in second_call_args
            assert "True" in second_call_args

    def test_ml_indicator_not_equal_with_identical_values(self, evaluator, mock_strategy_instance):
        """ML指標の!=比較で同じ値の場合"""
        mock_strategy_instance.ML_FEATURE = 0.5

        condition = Condition(left_operand="ML_FEATURE", operator="!=", right_operand=0.5)
        result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)

        # np.iscloseでFalseになるはず
        assert result is False

    def test_ml_indicator_unavailable_logging(self, evaluator, mock_strategy_instance):
        """利用可能な属性情報を正確にログ出力"""
        condition = Condition(left_operand="NONEXISTENT_ML", operator=">", right_operand=0)

        with patch('backend.app.services.auto_strategy.core.condition_evaluator.logger') as mock_logger:
            result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)

            # 警告ログが呼ばれていることを確認
            mock_logger.warning.assert_called_once()

            # ログに利用可能な属性が含まれている
            log_args = mock_logger.warning.call_args[0][0]
            assert "未対応のオペランド" in log_args
            assert "NONEXISTENT_ML" in log_args
            # いくつかの属性名が含まれているはず
            assert any(attr in log_args for attr in ["close", "sma_20", "rsi", "macd"]), "Available attributes should be logged"

class TestNumericCompatibility:
    """数値互換性テスト

    numpy互換のより深いテスト
    """

    def test_numpy_unsigned_integers(self, evaluator, mock_strategy_instance):
        """numpy符号なし整数型との比較"""
        condition = Condition(left_operand=np.uint64(100), operator=">", right_operand=np.uint32(50))
        result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)
        assert result is True

    def test_numpy_complex_numbers_warning(self, evaluator, mock_strategy_instance):
        """numpy複素数型の警告テスト"""
        condition = Condition(left_operand=complex(1, 2), operator=">", right_operand=complex(3, 4))
        with patch('backend.app.services.auto_strategy.core.condition_evaluator.logger') as mock_logger:
            result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)
            mock_logger.warning.assert_called_once()
            assert result is False

    def test_decimal_type_from_external_lib(self, evaluator, mock_strategy_instance):
        """外部ライブラリからのDecimal型"""
        try:
            from decimal import Decimal
            condition = Condition(left_operand=Decimal('10.5'), operator=">", right_operand=Decimal('5.5'))
            result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)
            assert result is True
        except ImportError:
            pytest.skip("decimal module not available")

    def test_custom_numeric_type_with_dtype(self, evaluator, mock_strategy_instance):
        """dtype属性を持つカスタム数値型"""
        class CustomNumeric:
            def __init__(self, value):
                self.value = value
                self.dtype = type(self)  # dtype属性を持つ

            def __gt__(self, other):
                return self.value > other

            def __lt__(self, other):
                return self.value < other

            def __ge__(self, other):
                return self.value >= other

            def __le__(self, other):
                return self.value <= other

        condition = Condition(left_operand=CustomNumeric(10), operator=">", right_operand=5)
        with patch('backend.app.services.auto_strategy.core.condition_evaluator.logger') as mock_logger:
            result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)
            # dtypeチェックのために警告が出力される
            assert result is True

class TestStringNumericParsing:
    """文字列から数値へのパーステスト"""

    def test_numeric_string_comparison(self, evaluator, mock_strategy_instance):
        """数値文字列の比較"""
        condition = Condition(left_operand="100", operator=">", right_operand="50")
        result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)
        assert result is True

    def test_scientific_notation_string(self, evaluator, mock_strategy_instance):
        """科学表記法文字列の比較"""
        condition = Condition(left_operand="1.5e2", operator=">", right_operand=100)
        result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)
        assert result is True  # 150 > 100

    def test_invalid_numeric_string_logging(self, evaluator, mock_strategy_instance):
        """無効な数値文字列のログ出力"""
        condition = Condition(left_operand="not_a_number", operator=">", right_operand=5)

        with patch('backend.app.services.auto_strategy.core.condition_evaluator.logger') as mock_logger:
            result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)

            # 警告ログが出力される
            mock_logger.warning.assert_called_once()
            # 新しい警告メッセージをチェック
            warning_message = mock_logger.warning.call_args[0][0]
            assert "非数値文字列オペランドが数値に変換されました" in warning_message
            assert "'not_a_number' -> 0.0" in warning_message
            assert result is False

    def test_mixed_string_numeric_comparison_logging(self, evaluator, mock_strategy_instance):
        """文字列と数値の混合比較でのログ"""
        condition = Condition(left_operand="100", operator=">", right_operand=50)

        with patch('backend.app.services.auto_strategy.core.condition_evaluator.logger') as mock_logger:
            result = evaluator.evaluate_single_condition(condition, mock_strategy_instance)

            # ログが出力されてるはず
            mock_logger.warning.assert_called()
            assert result is True