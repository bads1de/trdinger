"""レジーム一貫性ゲートのテスト"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.config.ga_config import (
    EvaluationConfig,
    ValidationConfig,
)
from app.services.auto_strategy.services.strategy_validation_service import (
    StrategyValidationService,
)


def _make_gene():
    return SimpleNamespace(
        indicators=[
            SimpleNamespace(
                type="BBANDS",
                parameters={"length": 20},
                enabled=True,
                timeframe=None,
                id="aaaa1111",
            )
        ],
        long_entry_conditions=[
            {
                "direction": "long",
                "left_operand": "close",
                "operator": ">",
                "right_operand": {"type": "indicator", "name": "BBANDS_aaaa1111_2"},
            }
        ],
        short_entry_conditions=[],
        long_exit_conditions=[],
        short_exit_conditions=[],
        risk_management={},
        tpsl_gene=None,
        position_sizing_gene=None,
        entry_gene=None,
        exit_gene=None,
        tool_genes=[],
    )


def _service_with_slice_responses(responses, captured=None):
    """スライス評価が responses を順に返すサービスを構築する。"""
    service = StrategyValidationService(MagicMock())
    iterator = iter(responses)

    def fake_eval(gene, bt_config, ga_config):
        if captured is not None:
            captured.append((bt_config.get("start_date"), bt_config.get("end_date")))
        return next(iterator)

    service._evaluate_variant_metrics = fake_eval
    return service


@pytest.fixture
def base():
    config = GAConfig(
        validation_config=ValidationConfig(
            regime_consistency_enabled=True,
            regime_splits=2,
            min_trades=10,
        ),
        evaluation_config=EvaluationConfig(evaluation_mode="walk_forward"),
        objectives=["total_return"],
    )
    bt = {"start_date": "2024-01-01", "end_date": "2025-01-01"}
    return config, bt


class TestRegimeConsistencyGate:
    """レジーム一貫性ゲートのテスト"""

    def test_all_slices_pass_keeps_passed(self, base):
        config, bt = base
        service = _service_with_slice_responses(
            [
                {"total_return": 0.05, "buy_hold_return": 0.03, "total_trades": 30},
                {"total_return": -0.02, "buy_hold_return": -0.05, "total_trades": 20},
            ]
        )
        validation = service._apply_regime_consistency_gate(
            {"passed": True, "reasons": []},
            _make_gene(),
            bt,
            config,
            config.validation_config,
        )

        assert validation["passed"] is True
        result = validation["regime_consistency"]
        assert result["splits"] == 2
        assert all(row["passed"] for row in result["slices"])

    def test_underperforming_slice_fails_closed(self, base):
        """片スライスでも市場比が負なら不合格"""
        config, bt = base
        service = _service_with_slice_responses(
            [
                # 前半: 市場比プラス
                {"total_return": 0.10, "buy_hold_return": 0.02, "total_trades": 30},
                # 後半: B&H +20% に対し戦略 +5% → 市場比マイナス
                {"total_return": 0.05, "buy_hold_return": 0.20, "total_trades": 25},
            ]
        )
        validation = service._apply_regime_consistency_gate(
            {"passed": True, "reasons": []},
            _make_gene(),
            bt,
            config,
            config.validation_config,
        )

        assert validation["passed"] is False
        assert any("レジーム一貫性なし" in r for r in validation["reasons"])

    def test_inactive_slice_by_trades_fails(self, base):
        """取引数がスライスあたり最低値未満のスライスがあれば不合格"""
        config, bt = base
        service = _service_with_slice_responses(
            [
                # スライスあたり必要取引数 = ceil(10/2) = 5
                {"total_return": 0.05, "buy_hold_return": 0.00, "total_trades": 3},
                {"total_return": 0.05, "buy_hold_return": 0.00, "total_trades": 30},
            ]
        )
        validation = service._apply_regime_consistency_gate(
            {"passed": True, "reasons": []},
            _make_gene(),
            bt,
            config,
            config.validation_config,
        )

        assert validation["passed"] is False

    def test_slices_cover_full_period(self, base):
        """スライスの日付が学習期間全体を隙間なくカバーする"""
        config, bt = base
        captured: list = []
        service = _service_with_slice_responses(
            [
                {"total_return": 0.1, "buy_hold_return": 0.0, "total_trades": 50},
                {"total_return": 0.1, "buy_hold_return": 0.0, "total_trades": 50},
            ],
            captured=captured,
        )
        service._apply_regime_consistency_gate(
            {"passed": True, "reasons": []},
            _make_gene(),
            bt,
            config,
            config.validation_config,
        )

        assert captured[0] == ("2024-01-01", "2024-07-02")
        assert captured[1] == ("2024-07-02", "2025-01-01")

    def test_disabled_gate_returns_validation_untouched(self):
        config = GAConfig(
            validation_config=ValidationConfig(regime_consistency_enabled=False),
            evaluation_config=EvaluationConfig(evaluation_mode="walk_forward"),
            objectives=["total_return"],
        )
        service = StrategyValidationService(MagicMock())
        original = {"passed": True, "reasons": []}
        validation = service._apply_regime_consistency_gate(
            original,
            _make_gene(),
            {"start_date": "2024-01-01", "end_date": "2025-01-01"},
            config,
            config.validation_config,
        )

        assert validation is original
        assert "regime_consistency" not in validation

    def test_empty_dates_skip_silently(self):
        """日付未指定の場合は判定をスキップする"""
        config = GAConfig(
            validation_config=ValidationConfig(regime_consistency_enabled=True),
            evaluation_config=EvaluationConfig(evaluation_mode="walk_forward"),
            objectives=["total_return"],
        )
        service = StrategyValidationService(MagicMock())
        original = {"passed": True, "reasons": []}
        validation = service._apply_regime_consistency_gate(
            original, _make_gene(), {}, config, config.validation_config
        )

        assert validation["passed"] is True
        assert "regime_consistency" not in validation
