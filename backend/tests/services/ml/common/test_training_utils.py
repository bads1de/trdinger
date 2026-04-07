"""
ML 学習用ユーティリティのテスト
"""

import pytest

from app.services.ml.common.training_utils import resolve_holdout_test_size


def test_resolve_holdout_test_size_prefers_explicit_test_size():
    assert resolve_holdout_test_size(
        test_size=0.3,
        train_test_split=0.8,
        validation_split=0.2,
    ) == pytest.approx(0.3)


def test_resolve_holdout_test_size_prefers_train_test_split_when_explicit():
    assert resolve_holdout_test_size(
        train_test_split=0.75,
        validation_split=0.2,
    ) == pytest.approx(0.25)


def test_resolve_holdout_test_size_uses_validation_split_when_train_split_default():
    assert resolve_holdout_test_size(
        train_test_split=0.8,
        validation_split=0.15,
    ) == pytest.approx(0.15)


def test_resolve_holdout_test_size_falls_back_to_default():
    assert resolve_holdout_test_size() == pytest.approx(0.2)
