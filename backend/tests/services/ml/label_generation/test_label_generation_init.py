"""
label_generationパッケージの__init__.pyのテスト

遅延ロード機能（__getattr__）とエクスポート定義を確認します。
"""

import pytest

import app.services.ml.label_generation as label_gen_package


class TestLabelGenerationInitExports:
    """label_generation/__init__.pyのエクスポートテスト"""

    def test_event_driven_label_generator_exported(self):
        """EventDrivenLabelGeneratorがエクスポートされている"""
        assert hasattr(label_gen_package, "EventDrivenLabelGenerator")

    def test_threshold_method_exported(self):
        """ThresholdMethodがエクスポートされている"""
        assert hasattr(label_gen_package, "ThresholdMethod")

    def test_barrier_profile_exported(self):
        """BarrierProfileがエクスポートされている"""
        assert hasattr(label_gen_package, "BarrierProfile")

    def test_get_common_presets_exported(self):
        """get_common_presetsがエクスポートされている"""
        assert hasattr(label_gen_package, "get_common_presets")

    def test_apply_preset_by_name_exported(self):
        """apply_preset_by_nameがエクスポートされている"""
        assert hasattr(label_gen_package, "apply_preset_by_name")

    def test_signal_generator_exported(self):
        """SignalGeneratorがエクスポートされている"""
        assert hasattr(label_gen_package, "SignalGenerator")

    def test_label_cache_exported(self):
        """LabelCacheがエクスポートされている"""
        assert hasattr(label_gen_package, "LabelCache")

    def test_label_generation_service_lazy_load(self):
        """LabelGenerationServiceが遅延ロードされる"""
        from app.services.ml.label_generation.label_generation_service import (
            LabelGenerationService,
        )

        service = getattr(label_gen_package, "LabelGenerationService")

        assert service is LabelGenerationService

    def test_getattr_raises_for_non_existent(self):
        """存在しない属性でAttributeErrorが発生する"""
        with pytest.raises(AttributeError, match="module.*has no attribute"):
            _ = label_gen_package.NonExistentAttribute

    def test_all_contains_expected_items(self):
        """__all__に期待されるアイテムが含まれる"""
        expected_items = [
            "EventDrivenLabelGenerator",
            "ThresholdMethod",
            "BarrierProfile",
            "get_common_presets",
            "apply_preset_by_name",
            "SignalGenerator",
            "LabelCache",
        ]

        for item in expected_items:
            assert item in label_gen_package.__all__, f"{item} not in __all__"

    def test_all_is_list(self):
        """__all__がリストである"""
        assert isinstance(label_gen_package.__all__, list)

    def test_module_has_docstring(self):
        """モジュールにドキュメント文字列がある"""
        assert label_gen_package.__doc__ is not None
        assert len(label_gen_package.__doc__) > 0
