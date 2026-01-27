import pytest
from unittest.mock import MagicMock, patch
from app.services.backtest.factories.auto_strategy_loader import (
    AutoStrategyLoader,
    AutoStrategyLoaderError,
)


class TestAutoStrategyLoader:
    @pytest.fixture
    def loader(self):
        return AutoStrategyLoader()

    def test_extract_strategy_gene(self, loader):
        # パターン1: 直接
        config1 = {"strategy_gene": {"id": 1}}
        assert loader._extract_strategy_gene(config1) == {"id": 1}

        # パターン2: parameters内
        config2 = {"parameters": {"strategy_gene": {"id": 2}}}
        assert loader._extract_strategy_gene(config2) == {"id": 2}

        # パターン3: なし
        assert loader._extract_strategy_gene({}) == {}

    def test_load_strategy_gene_success(self, loader):
        config = {"strategy_gene": {"id": "test"}}
        mock_gene = MagicMock()

        # StrategyGene must be a type for isinstance check
        class MockStrategyGene:
            pass

        # app.services.auto_strategy 関連のモジュールをモック
        with (
            patch(
                "app.services.auto_strategy.genes.StrategyGene", new=MockStrategyGene
            ),
            patch(
                "app.services.auto_strategy.serializers.serialization.GeneSerializer"
            ) as MockSerializerClass,
        ):

            MockSerializerClass.return_value.dict_to_strategy_gene.return_value = (
                mock_gene
            )

            result = loader.load_strategy_gene(config)

            assert result == mock_gene
            MockSerializerClass.return_value.dict_to_strategy_gene.assert_called_once()

    def test_load_strategy_gene_direct_object(self, loader):
        # StrategyGene must be a type for isinstance check
        class MockStrategyGene:
            pass

        gene_instance = MockStrategyGene()
        config = {"strategy_gene": gene_instance}

        with (
            patch(
                "app.services.auto_strategy.genes.StrategyGene", new=MockStrategyGene
            ),
            patch(
                "app.services.auto_strategy.serializers.serialization.GeneSerializer"
            ) as MockSerializerClass,
        ):

            result = loader.load_strategy_gene(config)

            assert result == gene_instance
            # Serialization should be skipped
            MockSerializerClass.return_value.dict_to_strategy_gene.assert_not_called()

    def test_load_strategy_gene_no_data(self, loader):
        with pytest.raises(AutoStrategyLoaderError) as excinfo:
            loader.load_strategy_gene({})
        assert "含まれていません" in str(excinfo.value)

    def test_load_strategy_gene_import_error(self, loader):
        # インポート時にエラーが出るようにパッチ（少しトリッキー）
        # load_strategy_geneの先頭でインポートしている箇所をターゲットにする
        with patch("builtins.__import__", side_effect=ImportError("Module not found")):
            with pytest.raises(AutoStrategyLoaderError) as excinfo:
                loader.load_strategy_gene({"strategy_gene": {}})
            assert "インポートに失敗しました" in str(excinfo.value)

    def test_create_auto_strategy_class_success(self, loader):
        config = {"strategy_gene": {}}
        mock_gene = MagicMock()
        mock_gene.validate.return_value = (True, [])

        from app.services.auto_strategy.strategies.universal_strategy import (
            UniversalStrategy,
        )

        with (
            patch.object(loader, "load_strategy_gene", return_value=mock_gene),
            patch(
                "app.services.auto_strategy.strategies.universal_strategy.UniversalStrategy",
                UniversalStrategy,
            ),
        ):

            result = loader.create_auto_strategy_class(config)

            assert result == UniversalStrategy

    def test_create_auto_strategy_class_invalid_gene(self, loader):
        config = {"strategy_gene": {}}
        mock_gene = MagicMock()
        mock_gene.validate.return_value = (False, ["Error 1", "Error 2"])

        with patch.object(loader, "load_strategy_gene", return_value=mock_gene):
            with pytest.raises(AutoStrategyLoaderError) as excinfo:
                loader.create_auto_strategy_class(config)
            assert "無効な戦略遺伝子です" in str(excinfo.value)
            assert "Error 1" in str(excinfo.value)
