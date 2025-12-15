"""
Bybitサービス基底クラスのテスト

BybitServiceクラスの全機能をテストします:
- 初期化とCCXT設定
- パラメータ検証
- CCXTエラーハンドリング
- ページネーション処理
- データベースセッション管理
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import ccxt
import pytest

from app.services.data_collection.bybit.bybit_service import BybitService
from app.utils.error_handler import DataError


class ConcreteBybitService(BybitService):
    """テスト用の具象クラス"""

    pass


@pytest.fixture
def mock_exchange():
    """モックCCXT取引所"""
    exchange = MagicMock(spec=ccxt.bybit)
    exchange.fetch_funding_rate_history = MagicMock()
    exchange.fetch_open_interest_history = MagicMock()
    return exchange


@pytest.fixture
def service(mock_exchange):
    """サービスインスタンス"""
    with patch("app.services.data_collection.bybit.bybit_service.ccxt.bybit") as mock:
        mock.return_value = mock_exchange
        return ConcreteBybitService()


@pytest.fixture
def mock_config():
    """モック設定"""
    config = MagicMock()
    config.max_limit = 1000
    config.bybit_page_limit = 200
    config.bybit_max_pages = 50
    return config


@pytest.mark.asyncio
class TestServiceInitialization:
    """サービス初期化テスト"""

    async def test_service_initialization(self):
        """サービスが正しく初期化されることを確認"""
        with patch(
            "app.services.data_collection.bybit.bybit_service.ccxt.bybit"
        ) as mock:
            mock_exchange = MagicMock()
            mock.return_value = mock_exchange

            service = ConcreteBybitService()

            assert service.exchange is not None
            mock.assert_called_once_with(
                {
                    "sandbox": False,
                    "enableRateLimit": True,
                    "options": {"defaultType": "linear"},
                }
            )

    async def test_exchange_configuration(self, service):
        """取引所設定が正しく適用されていることを確認"""
        assert hasattr(service, "exchange")


@pytest.mark.asyncio
class TestParameterValidation:
    """パラメータ検証テスト"""

    async def test_validate_symbol_valid(self, service):
        """有効なシンボルが検証を通過することを確認"""
        service._validate_symbol("BTC/USDT:USDT")
        service._validate_symbol("BTC/USDT:USDT")

    async def test_validate_symbol_invalid_empty(self, service):
        """空のシンボルがValueErrorを発生させることを確認"""
        with pytest.raises(
            ValueError, match="シンボルは有効な文字列である必要があります"
        ):
            service._validate_symbol("")

    async def test_validate_symbol_invalid_none(self, service):
        """Noneシンボルが ValueErrorを発生させることを確認"""
        with pytest.raises(
            ValueError, match="シンボルは有効な文字列である必要があります"
        ):
            service._validate_symbol(None)

    async def test_validate_symbol_invalid_type(self, service):
        """無効な型のシンボルがValueErrorを発生させることを確認"""
        with pytest.raises(
            ValueError, match="シンボルは有効な文字列である必要があります"
        ):
            service._validate_symbol(123)

    @patch("app.services.data_collection.bybit.bybit_service.unified_config")
    async def test_validate_limit_valid(self, mock_unified_config, service):
        """有効なlimitが検証を通過することを確認"""
        mock_unified_config.data_collection.max_limit = 1000
        service._validate_limit(100)
        service._validate_limit(1000)
        service._validate_limit(1)

    @patch("app.services.data_collection.bybit.bybit_service.unified_config")
    async def test_validate_limit_invalid_zero(self, mock_unified_config, service):
        """ゼロのlimitがValueErrorを発生させることを確認"""
        mock_unified_config.data_collection.max_limit = 1000
        with pytest.raises(
            ValueError, match="limitは1から1000の間の整数である必要があります"
        ):
            service._validate_limit(0)

    @patch("app.services.data_collection.bybit.bybit_service.unified_config")
    async def test_validate_limit_invalid_negative(self, mock_unified_config, service):
        """負のlimitがValueErrorを発生させることを確認"""
        mock_unified_config.data_collection.max_limit = 1000
        with pytest.raises(
            ValueError, match="limitは1から1000の間の整数である必要があります"
        ):
            service._validate_limit(-1)

    @patch("app.services.data_collection.bybit.bybit_service.unified_config")
    async def test_validate_limit_invalid_exceed(self, mock_unified_config, service):
        """最大値を超えるlimitがValueErrorを発生させることを確認"""
        mock_unified_config.data_collection.max_limit = 1000
        with pytest.raises(
            ValueError, match="limitは1から1000の間の整数である必要があります"
        ):
            service._validate_limit(1001)

    @patch("app.services.data_collection.bybit.bybit_service.unified_config")
    async def test_validate_limit_none(self, mock_unified_config, service):
        """Noneのlimitが検証を通過することを確認"""
        mock_unified_config.data_collection.max_limit = 1000
        service._validate_limit(None)

    @patch("app.services.data_collection.bybit.bybit_service.unified_config")
    async def test_validate_parameters_valid(self, mock_unified_config, service):
        """有効なパラメータが検証を通過することを確認"""
        mock_unified_config.data_collection.max_limit = 1000
        service._validate_parameters("BTC/USDT:USDT", 100)

    @patch("app.services.data_collection.bybit.bybit_service.unified_config")
    async def test_validate_parameters_invalid_symbol(
        self, mock_unified_config, service
    ):
        """無効なシンボルがValueErrorを発生させることを確認"""
        mock_unified_config.data_collection.max_limit = 1000
        with pytest.raises(ValueError):
            service._validate_parameters("", 100)

    @patch("app.services.data_collection.bybit.bybit_service.unified_config")
    async def test_validate_parameters_invalid_limit(
        self, mock_unified_config, service
    ):
        """無効なlimitがValueErrorを発生させることを確認"""
        mock_unified_config.data_collection.max_limit = 1000
        with pytest.raises(ValueError):
            service._validate_parameters("BTC/USDT:USDT", 0)


@pytest.mark.asyncio
class TestCCXTErrorHandling:
    """CCXTエラーハンドリングテスト"""

    async def test_handle_ccxt_errors_success(self, service, mock_exchange):
        """正常なCCXT操作が成功することを確認"""
        test_func = AsyncMock(return_value=["data"])

        result = await service._handle_ccxt_errors_impl(
            "テスト操作", test_func, "arg1", "arg2"
        )

        assert result == ["data"]
        test_func.assert_called_once_with("arg1", "arg2")

    async def test_handle_ccxt_errors_bad_symbol(self, service, mock_exchange):
        """BadSymbolエラーがDataErrorに変換されることを確認"""
        test_func = MagicMock(side_effect=ccxt.BadSymbol("Invalid symbol"))

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_executor = AsyncMock(side_effect=ccxt.BadSymbol("Invalid symbol"))
            mock_loop.return_value.run_in_executor = mock_executor

            with pytest.raises(DataError, match="無効なシンボル"):
                await service._handle_ccxt_errors_impl("テスト操作", test_func)

    async def test_handle_ccxt_errors_network_error(self, service, mock_exchange):
        """NetworkErrorがDataErrorに変換されることを確認"""
        test_func = MagicMock(side_effect=ccxt.NetworkError("Connection failed"))

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_executor = AsyncMock(
                side_effect=ccxt.NetworkError("Connection failed")
            )
            mock_loop.return_value.run_in_executor = mock_executor

            with pytest.raises(DataError, match="ネットワークエラー"):
                await service._handle_ccxt_errors_impl("テスト操作", test_func)

    async def test_handle_ccxt_errors_exchange_error(self, service, mock_exchange):
        """ExchangeErrorがDataErrorに変換されることを確認"""
        test_func = MagicMock(side_effect=ccxt.ExchangeError("Exchange error"))

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_executor = AsyncMock(side_effect=ccxt.ExchangeError("Exchange error"))
            mock_loop.return_value.run_in_executor = mock_executor

            with pytest.raises(DataError, match="取引所エラー"):
                await service._handle_ccxt_errors_impl("テスト操作", test_func)

    async def test_handle_ccxt_errors_generic_exception(self, service, mock_exchange):
        """一般的な例外がDataErrorに変換されることを確認"""
        test_func = MagicMock(side_effect=Exception("Unexpected error"))

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_executor = AsyncMock(side_effect=Exception("Unexpected error"))
            mock_loop.return_value.run_in_executor = mock_executor

            with pytest.raises(DataError):
                await service._handle_ccxt_errors_impl("テスト操作", test_func)


@pytest.mark.asyncio
class TestHelperMethods:
    """ヘルパーメソッドテスト"""

    async def test_get_interval_milliseconds_5min(self, service):
        """5分間隔のミリ秒変換を確認"""
        result = service._get_interval_milliseconds("5min")
        assert result == 5 * 60 * 1000

    async def test_get_interval_milliseconds_1h(self, service):
        """1時間間隔のミリ秒変換を確認"""
        result = service._get_interval_milliseconds("1h")
        assert result == 60 * 60 * 1000

    async def test_get_interval_milliseconds_1d(self, service):
        """1日間隔のミリ秒変換を確認"""
        result = service._get_interval_milliseconds("1d")
        assert result == 24 * 60 * 60 * 1000

    async def test_get_interval_milliseconds_unknown(self, service):
        """不明な間隔でデフォルト値（1時間）を返すことを確認"""
        result = service._get_interval_milliseconds("unknown")
        assert result == 60 * 60 * 1000

    async def test_convert_to_api_symbol_basic(self, service):
        """基本的なシンボル変換を確認"""
        result = service._convert_to_api_symbol("BTC/USDT:USDT")
        assert result == "BTCUSDT"

    async def test_convert_to_api_symbol_no_colon(self, service):
        """コロンなしシンボルの変換を確認"""
        result = service._convert_to_api_symbol("BTC/USDT:USDT")
        assert result == "BTCUSDT"

    async def test_convert_to_api_symbol_already_converted(self, service):
        """既に変換済みのシンボルを確認"""
        result = service._convert_to_api_symbol("BTCUSDT")
        assert result == "BTCUSDT"

    async def test_convert_to_api_symbol_non_string(self, service):
        """非文字列入力の変換を確認"""
        result = service._convert_to_api_symbol(123)
        assert result == "123"


@pytest.mark.asyncio
class TestProcessPageData:
    """ページデータ処理テスト"""

    async def test_process_page_data_no_duplicates(self, service):
        """重複なしのページデータ処理を確認"""
        page_data = [
            {"timestamp": 1000},
            {"timestamp": 2000},
            {"timestamp": 3000},
        ]
        all_data = []

        result = service._process_page_data(page_data, all_data, None, 1)

        assert result == page_data
        assert len(result) == 3

    async def test_process_page_data_with_duplicates(self, service):
        """重複ありのページデータ処理を確認"""
        page_data = [
            {"timestamp": 1000},
            {"timestamp": 2000},
            {"timestamp": 3000},
        ]
        all_data = [{"timestamp": 2000}]

        result = service._process_page_data(page_data, all_data, None, 1)

        assert len(result) == 2
        assert {"timestamp": 2000} not in result

    async def test_process_page_data_incremental_update(self, service):
        """差分更新でのページデータ処理を確認"""
        page_data = [
            {"timestamp": 3000},
            {"timestamp": 4000},
            {"timestamp": 5000},
        ]
        all_data = []
        latest_existing_timestamp = 4500

        result = service._process_page_data(
            page_data, all_data, latest_existing_timestamp, 1
        )

        assert len(result) == 2
        assert all(item["timestamp"] < latest_existing_timestamp for item in result)

    async def test_process_page_data_incremental_complete(self, service):
        """差分更新完了時の処理を確認"""
        page_data = [
            {"timestamp": 1000},
            {"timestamp": 2000},
        ]
        all_data = []
        latest_existing_timestamp = 500

        result = service._process_page_data(
            page_data, all_data, latest_existing_timestamp, 1
        )

        assert result is None

    async def test_process_page_data_open_interest_incremental(self, service):
        """オープンインタレスト差分更新を確認"""
        page_data = [
            {"timestamp": 5000},
            {"timestamp": 6000},
            {"timestamp": 7000},
        ]
        all_data = []
        latest_existing_timestamp = 5500

        result = service._process_page_data(
            page_data, all_data, latest_existing_timestamp, 1, "open_interest"
        )

        assert len(result) == 2
        assert all(item["timestamp"] > latest_existing_timestamp for item in result)


@pytest.mark.asyncio
class TestDatabaseSessionManagement:
    """データベースセッション管理テスト"""

    async def test_execute_with_db_session_no_repository(self, service):
        """リポジトリなしでのDB操作を確認"""
        mock_func = AsyncMock(return_value="result")

        with patch(
            "app.services.data_collection.bybit.bybit_service.get_db"
        ) as mock_get_db:
            mock_db = MagicMock()
            mock_get_db.return_value = iter([mock_db])

            result = await service._execute_with_db_session(mock_func)

            assert result == "result"
            mock_func.assert_called_once()
            mock_db.close.assert_called_once()

    async def test_execute_with_db_session_with_repository(self, service):
        """リポジトリありでのDB操作を確認"""
        mock_func = AsyncMock(return_value="result")
        mock_repo = MagicMock()

        result = await service._execute_with_db_session(mock_func, repository=mock_repo)

        assert result == "result"
        mock_func.assert_called_once_with(db=None, repository=mock_repo)

    async def test_get_latest_timestamp_from_db(self, service):
        """最新タイムスタンプ取得を確認"""
        mock_repo_class = MagicMock()
        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo

        test_datetime = datetime(2023, 1, 1, 0, 0, 0)
        mock_repo.get_latest_timestamp.return_value = test_datetime

        with patch(
            "app.services.data_collection.bybit.bybit_service.get_db"
        ) as mock_get_db:
            mock_db = MagicMock()
            mock_get_db.return_value = iter([mock_db])

            result = await service._get_latest_timestamp_from_db(
                mock_repo_class, "get_latest_timestamp", "BTC/USDT:USDT"
            )

            expected_timestamp = int(test_datetime.timestamp() * 1000)
            assert result == expected_timestamp

    async def test_get_latest_timestamp_from_db_no_data(self, service):
        """データなし時の最新タイムスタンプ取得を確認"""
        mock_repo_class = MagicMock()
        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_latest_timestamp.return_value = None

        with patch(
            "app.services.data_collection.bybit.bybit_service.get_db"
        ) as mock_get_db:
            mock_db = MagicMock()
            mock_get_db.return_value = iter([mock_db])

            result = await service._get_latest_timestamp_from_db(
                mock_repo_class, "get_latest_timestamp", "BTC/USDT:USDT"
            )

            assert result is None

    async def test_get_latest_timestamp_from_db_error(self, service):
        """エラー時の最新タイムスタンプ取得を確認"""
        mock_repo_class = MagicMock()

        with patch(
            "app.services.data_collection.bybit.bybit_service.get_db"
        ) as mock_get_db:
            mock_get_db.side_effect = Exception("Database error")

            result = await service._get_latest_timestamp_from_db(
                mock_repo_class, "get_latest_timestamp", "BTC/USDT:USDT"
            )

            assert result is None




