"""
RiskRewardTPSLGenerator のユニットテスト（TDD: 先に失敗するテストを追加）
"""

import pytest

from app.services.auto_strategy.generators.risk_reward_tpsl_generator import (
    RiskRewardTPSLGenerator,
    RiskRewardConfig,
    RiskRewardProfile,
)


def test_generate_risk_reward_basic():
    generator = RiskRewardTPSLGenerator()
    config = RiskRewardConfig(target_ratio=2.0)

    result = generator.generate_risk_reward_tpsl(stop_loss_pct=0.03, config=config)

    assert pytest.approx(result.take_profit_pct, rel=1e-6) == 0.06
    assert result.actual_risk_reward_ratio == pytest.approx(2.0, rel=1e-6)
    assert result.is_ratio_achieved is True


def test_generate_risk_reward_respects_limits():
    generator = RiskRewardTPSLGenerator()
    # 大きすぎるTPを上限にクリップ
    config = RiskRewardConfig(target_ratio=20.0, max_tp_limit=0.2)
    result = generator.generate_risk_reward_tpsl(stop_loss_pct=0.03, config=config)

    assert result.take_profit_pct == pytest.approx(0.2, rel=1e-6)
    assert result.actual_risk_reward_ratio == pytest.approx(0.2 / 0.03, rel=1e-6)
    assert result.is_ratio_achieved is False


def test_profile_default_ratio_helper():
    generator = RiskRewardTPSLGenerator()
    ratio = generator.get_recommended_ratio_for_profile(RiskRewardProfile.BALANCED)
    assert 1.5 <= ratio <= 3.0

