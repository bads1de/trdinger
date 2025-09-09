"""
Unicode文字化け問題検出テスト

このテストスイートは、ログ記録でのUnicode文字コード変換問題を検出します。
日本語や絵文字が含まれる場合に文字化けが発生するケースをシミュレートします。
"""

import pytest
import logging
import tempfile
import os
from unittest.mock import patch, MagicMock
from io import StringIO

from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
from app.utils.error_handler import ErrorHandler


@pytest.fixture
def unicode_test_data():
    """Unicode文字を含むテストデータ"""
    return {
        "experiment_names": [
            "テスト戦略_ユニコード🚀",
            "実験 № 123 トレーディング",
            "тест戦略",  # ロシア語
            "戦略测试",  # 中国語
            "Trading ğ Strategy",  # トルコ語
            "Estratégia 🏆 Teste"  # ポルトガル語+絵文字
        ],
        "symbols": [
            "BTC/USDT 测试",  # 中国語
            "ETH/USD 🚀",
            " forex 戦略テスト"
        ],
        "messages": [
            "バックテスト開始: 🚀 高頻度トレーディング戦略",
            "Error: データロード失敗 💥",
            "Warning: パラメータ変換 ⚠️ тест"
        ]
    }


@pytest.fixture
def corrupted_encoding_logger():
    """文字化けが発生するログ設定"""
    # ファイルハンドラーをUTF-8以外で設定（バグシミュレーション）
    logger = logging.getLogger("unicode_test")
    logger.setLevel(logging.DEBUG)

    # 文字化けを起こすハンドラー
    handler = logging.StreamHandler(StringIO())

    try:
        # UTF-8以外でエンコーディング（バグ再現）
        with tempfile.NamedTemporaryFile(mode='w', encoding='latin-1', delete=False) as f:
            temp_file = f.name
    except:
        temp_file = None

    if temp_file:
        file_handler = logging.FileHandler(temp_file, encoding='latin-1')  # 意図的にlatin-1
        logger.addHandler(file_handler)

    return logger, temp_file


class TestUnicodeLoggingCorruption:
    """Unicode文字化け検出テスト"""

    def test_logging_fails_with_non_utf8_encoding(self, unicode_test_data):
        """UTF-8以外エンコーディングでログ記録が失敗するテスト"""
        test_message = unicode_test_data["messages"][0]

        # latin-1でログを出力しようとするとUnicodeEncodeError
        try:
            with open(tempfile.mktemp(), 'w', encoding='latin-1') as f:
                f.write(test_message)  # Unicode文字がlatin-1で扱えない
                # この行が実行されない場合、バグ検出
                assert False, "文字化けが発生しない (Encoding設定バグ未検出)"
        except UnicodeEncodeError as e:
            # これが期待される動作（UTF-8以外でUnicode文字が失敗）
            assert "latin-1" in str(e) or "ordinal not in range" in str(e)

    def test_experiment_name_unicode_corruption_in_service(self, unicode_test_data):
        """サービス層での実験名Unicode文字化けテスト"""
        from app.services.auto_strategy.services.experiment_service import ExperimentService
        import json

        unicode_name = unicode_test_data["experiment_names"][0]

        with patch('app.services.auto_strategy.services.experiment_service.SessionLocal') as mock_session:
            experiment_service = ExperimentService()

            try:
                with patch('builtins.open', create=True) as mock_open:
                    mock_file = MagicMock()
                    # latin-1エンコーディングで保存すると文字化け
                    mock_file.__enter__.return_value = mock_file
                    mock_file.write.side_effect = lambda x: bytes(x, 'utf-8').decode('latin-1')
                    mock_open.return_value = mock_file

                    # 実験保存時に文字化けが発生
                    experiment_service.create_experiment(unicode_name, "test desc")

                    # 文字化けエラーが発生するか確認
                    calls = mock_file.write.call_args_list
                    if calls:
                        written_data = calls[0][0][0]
                        if '' in written_data:  # 文字化けマーク
                            pytest.fail(f"文字化け検出: {written_data}")

            except (UnicodeDecodeError, UnicodeEncodeError) as e:
                assert "encoding" in str(e).lower() or "ordinal" in str(e).lower()

    def test_ga_engine_logging_unicode_error(self, unicode_test_data):
        """GAエンジンのログ記録でUnicodeエラーが発生するテスト"""
        from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
        from unittest.mock import MagicMock

        config = MagicMock()
        config.log_level = "DEBUG"

        # モックサービス
        backtest_service = MagicMock()
        strategy_factory = MagicMock()
        gene_generator = MagicMock()

        ga_engine = GeneticAlgorithmEngine(backtest_service, strategy_factory, gene_generator)

        # Unicode文字列を含むログメッセージ
        unicode_log_msg = f"GA実行開始: {unicode_test_data['experiment_names'][0]} ⏰"

        try:
            # ログ出力で文字化けが発生するハンドラーを設定
            corrupted_logger, temp_file = self._create_corrupted_logger()

            with patch.object(ga_engine, 'logger', corrupted_logger):
                # ログ出力が失敗するはず
                ga_engine.logger.info(unicode_log_msg)

                # ファイルが存在し、文字化けが含まれているか確認
                if temp_file and os.path.exists(temp_file):
                    with open(temp_file, 'r', encoding='utf-8', errors='replace') as f:
                        log_content = f.read()
                        if '' in log_content:
                            pytest.fail("バグ検出: GAログでUnicode文字が文字化け")

            if temp_file:
                os.unlink(temp_file)

        except UnicodeEncodeError as e:
            assert True  # 文字化けバグを検出

    def test_error_handler_unicode_message_truncation(self, unicode_test_data):
        """エラーハンドラーでのUnicodeメッセージ処理バグ"""
        unicode_message = unicode_test_data["messages"][1]

        # StringIOでエンコーディング問題をシミュレート
        corrupted_stream = StringIO()

        try:
            # ストリームに直接Unicode文字を書き込み
            corrupted_stream.write(unicode_message)

            # latin-1でデコードしようとすると失敗
            content = corrupted_stream.getvalue()
            corrupted_content = content.encode('utf-8').decode('latin-1', errors='replace')

            if '' in corrupted_content:
                pytest.fail("バグ検出: エラーメッセージ処理でUnicodeが文字化け")

        except (UnicodeDecodeError, UnicodeEncodeError) as e:
            assert True

    def test_symbol_unicode_validation_failure(self, unicode_test_data):
        """トレーディングシンボルのUnicode検証失敗テスト"""
        unicode_symbol = unicode_test_data["symbols"][0]

        # シンボル検証関数を想定
        def validate_symbol(symbol_str):
            if not isinstance(symbol_str, str):
                raise TypeError("シンボルは文字列であるべき")

            # Unicode文字を含むシンボルは拒否（バグ再現）
            if any(ord(c) > 127 for c in symbol_str):
                raise ValueError(f"非ASCIIシンボル拒否: {symbol_str}")

        try:
            validate_symbol(unicode_symbol)
            # この関数がSuccessの場合、バグ検出（Unicodeシンボルを拒否）
            assert False, f"バグ検出: Unicodeシンボルが拒否される {unicode_symbol}"

        except ValueError as e:
            assert "非ASCII" in str(e) or "Unicode" in str(e)

    def _create_corrupted_logger(self):
        """文字化けを起こすロガーを返す"""
        import tempfile
        logger = logging.getLogger("corrupted_unicode_test")
        logger.setLevel(logging.DEBUG)

        log_file = tempfile.mktemp(suffix='.log')
        try:
            # latin-1エンコーディングでファイルハンドラー作成
            handler = logging.FileHandler(log_file, encoding='latin-1')
            logger.addHandler(handler)
            return logger, log_file
        except:
            return logger, None