"""
auto_strategy/configパッケージの__init__.pyのテスト

遅延ロード機能（__getattr__）とエクスポート定義を確認します。
"""

import importlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import app.services.auto_strategy.config as config_package


class TestAutoStrategyConfigInitExports:
    """auto_strategy/config/__init__.pyのエクスポートテスト"""

    def test_base_config_is_not_exported(self):
        """BaseConfig は削除済みで公開しない"""
        assert hasattr(config_package, "BaseConfig") is False

    def test_auto_strategy_config_is_not_exported(self):
        """AutoStrategyConfig は削除済みで公開しない"""
        assert hasattr(config_package, "AutoStrategyConfig") is False

    def test_mutation_config_exported(self):
        """MutationConfigがエクスポートされている"""
        assert hasattr(config_package, "MutationConfig")

    def test_evaluation_config_exported(self):
        """EvaluationConfigがエクスポートされている"""
        assert hasattr(config_package, "EvaluationConfig")

    def test_hybrid_config_exported(self):
        """HybridConfigがエクスポートされている"""
        assert hasattr(config_package, "HybridConfig")

    def test_tuning_config_exported(self):
        """TuningConfigがエクスポートされている"""
        assert hasattr(config_package, "TuningConfig")

    def test_two_stage_selection_config_exported(self):
        """TwoStageSelectionConfigがエクスポートされている"""
        assert hasattr(config_package, "TwoStageSelectionConfig")

    def test_robustness_config_exported(self):
        """RobustnessConfigがエクスポートされている"""
        assert hasattr(config_package, "RobustnessConfig")

    def test_early_termination_settings_exported(self):
        """EarlyTerminationSettingsがエクスポートされている"""
        assert hasattr(config_package, "EarlyTerminationSettings")

    def test_ga_config_lazy_load(self):
        """GAConfigが遅延ロードされる"""
        from app.services.auto_strategy.config.ga import GAConfig

        config = getattr(config_package, "GAConfig")

        assert config is GAConfig

    def test_pyright_resolves_lazy_ga_exports(self):
        """pyright が GA 関連の遅延 export を正しく解決できる"""
        backend_root = Path(__file__).resolve().parents[4]
        snippet_path = backend_root / "tests" / "_pyright_lazy_ga_exports.py"
        try:
            snippet_path.write_text(
                textwrap.dedent(
                    """
                    from app.services.auto_strategy.config import (
                        ConfigValidator,
                        GAConfig,
                    )

                    def build_config() -> GAConfig:
                        config = GAConfig.from_dict({})
                        ConfigValidator.validate(config)
                        return config
                    """
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, "-m", "pyright", str(snippet_path)],
                cwd=backend_root,
                check=False,
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, result.stdout
        finally:
            snippet_path.unlink(missing_ok=True)

    def test_getattr_raises_for_non_existent(self):
        """存在しない属性でAttributeErrorが発生する"""
        with pytest.raises(AttributeError, match="module.*has no attribute"):
            _ = config_package.NonExistentAttribute

    def test_all_contains_expected_items(self):
        """__all__に期待されるアイテムが含まれる"""
        expected_items = [
            "EarlyTerminationSettings",
            "MutationConfig",
            "EvaluationConfig",
            "HybridConfig",
            "TuningConfig",
            "TwoStageSelectionConfig",
            "RobustnessConfig",
        ]

        for item in expected_items:
            assert item in config_package.__all__, f"{item} not in __all__"

    def test_auto_strategy_settings_module_is_removed(self):
        """旧 auto_strategy_settings モジュールは削除済み"""
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(
                "app.services.auto_strategy.config.auto_strategy_settings"
            )

    def test_all_is_list(self):
        """__all__がリストである"""
        assert isinstance(config_package.__all__, list)

    def test_module_has_docstring(self):
        """モジュールにドキュメント文字列がある"""
        assert config_package.__doc__ is not None
        assert len(config_package.__doc__) > 0

    def test_ga_nested_configs_module_exports_runtime_configs(self):
        """意味が明確な新モジュール名から runtime config 群を読める"""
        nested_configs = importlib.import_module(
            "app.services.auto_strategy.config.ga.nested_configs"
        )

        assert nested_configs.MutationConfig is config_package.MutationConfig
        assert nested_configs.EvaluationConfig is config_package.EvaluationConfig
        assert nested_configs.HybridConfig is config_package.HybridConfig
        assert (
            nested_configs.TwoStageSelectionConfig
            is config_package.TwoStageSelectionConfig
        )

    def test_legacy_base_module_is_removed(self):
        """旧 base モジュールは削除済み"""
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("app.services.auto_strategy.config.base")

    def test_internal_code_no_longer_references_legacy_nested_config_module(self):
        """コードベース内部では旧 sub_configs モジュールを参照しない"""
        project_root = Path(__file__).resolve().parents[4]
        current_file = Path(__file__).resolve()
        removed_legacy_module_file = (
            project_root
            / "app"
            / "services"
            / "auto_strategy"
            / "config"
            / "sub_configs.py"
        )
        assert removed_legacy_module_file.exists() is False

        legacy_module_name = "_".join(("sub", "configs"))
        forbidden_patterns = (
            ".".join(("config", legacy_module_name)),
            "." + legacy_module_name + " import",
        )

        offenders = []
        for root_name in ("app", "tests"):
            root_dir = project_root / root_name
            for file_path in root_dir.rglob("*.py"):
                if file_path == current_file:
                    continue

                content = file_path.read_text(encoding="utf-8")
                if any(pattern in content for pattern in forbidden_patterns):
                    offenders.append(str(file_path.relative_to(project_root)))

        assert offenders == []
