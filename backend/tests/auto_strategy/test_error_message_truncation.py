"""
エラーメッセージ切断問題検出テスト

このテストスイートは、長いエラーメッセージが途中で切断される問題を検出します。
ログ出力、APIレスポンス、エラーハンドリングでのメッセージ限界テストをします。
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from fastapi import HTTPException


class TestErrorMessageTruncation:
    """エラーメッセージ切断バグ検出テスト"""

    def test_long_error_message_gets_truncated_in_response(self):
        """APIレスポンスで長いエラーメッセージが切断されるテスト"""
        from app.utils.error_handler import ErrorHandler

        # 非常に長いエラーメッセージ（1024文字以上）
        long_error_message = "エラー発生: データベース接続失敗 - サーバーとの通信がタイムアウトしました。" * 50

        try:
            # HTTPExceptionを生成
            exception = ErrorHandler.handle_api_error(
                ValueError(long_error_message),
                context="テストコンテキスト",
                status_code=500,
                error_code="TEST_ERROR"
            )

            # レスポンス詳細を取得
            response_detail = exception.detail

            # メッセージが含まれているか確認
            if isinstance(response_detail, dict):
                message_content = response_detail.get("message", "")
            else:
                message_content = str(response_detail)

            # 長さが制限されている場合、バグ検出
            max_expected_length = len(long_error_message) + 100  # 余裕を持たせて

            if len(str(message_content)) < len(long_error_message):
                pytest.fail(f"バグ検出: エラーメッセージが切断されました: {len(message_content)}/{len(long_error_message)}文字")

        except Exception as e:
            # 処理自体が失敗する場合もバグ
            pytest.fail(f"バグ検出: エラーメッセージ処理でエラー {e}")

    def test_nested_exception_message_truncation(self):
        """ネストした例外メッセージの切断テスト"""
        original_error = "オリジナルエラー: バックテスト実行時のメモリ不足 - 詳細情報: " * 20
        context_msg = "コンテキスト情報: 戦略最適化フェーズでの失敗 - " * 10

        try:
            # ネストした例外チェーン
            try:
                try:
                    raise MemoryError(original_error)
                except MemoryError as e:
                    raise RuntimeError(f"{context_msg} 内部エラー: {e}") from e
            except RuntimeError as e:
                raise ValueError(f"トップレベルエラー: {e}") from e

        except ValueError as top_error:
            error_message = str(top_error)

            # メッセージが完全かどうかチェック
            if len(error_message) < len(original_error) + len(context_msg):
                pytest.fail("バグ検出: ネストした例外メッセージが切断されました")

    def test_json_serialization_truncates_long_messages(self):
        """JSONシリアライズで長いメッセージが切断されるテスト"""
        # 非常に長いメッセージ
        long_details = {
            "error_type": "VALIDATION_ERROR",
            "error_message": "これは非常に長いエラーメッセージです。" * 200,
            "stack_trace": "Traceback (most recent call last):" * 50,
            "context": {
                "user_id": "test_user_12345",
                "operation": "strategy_optimization_test",
                "parameters": {"param_" + str(i): f"value_{i}_long_description" for i in range(100)}
            }
        }

        # JSONシリアライズ
        try:
            json_str = json.dumps(long_details, ensure_ascii=False, indent=2)

            # シリアライズ後にデシリアライズ
            parsed_back = json.loads(json_str)

            # 元のメッセージが完全に保持されているか確認
            original_msg_len = len(long_details["error_message"])
            restored_msg_len = len(parsed_back["error_message"])

            if restored_msg_len < original_msg_len:
                pytest.fail(f"バグ検出: JSONシリアライズでメッセージが切断されました {restored_msg_len}/{original_msg_len}")

        except (json.JSONDecodeError, ValueError) as e:
            pytest.fail(f"バグ検出: JSON処理でメッセージ切断エラー {e}")

    def test_error_handler_max_message_length_limit_bug(self):
        """エラーハンドラーのメッセージ長制限バグテスト"""
        # エラーハンドラーが暗黙的にメッセージ長を制限している場合
        MAX_MESSAGE_LENGTH = 500  # 仮定の制限

        very_long_message = "A" * 1000  # 制限を超える長さ

        with patch('builtins.print') as mock_print:
            try:
                # エラーハンドラーを使用
                from app.utils.error_handler import ErrorHandler

                ErrorHandler.handle_model_error(
                    ValueError(very_long_message),
                    context="test_context"
                )
            except Exception:
                pass

            # 出力されたログを確認
            logged_messages = [call[0][0] for call in mock_print.call_args_list if call[0]]

            for msg in logged_messages:
                if len(msg) < len(very_long_message) and len(msg) >= MAX_MESSAGE_LENGTH:
                    pytest.fail("バグ検出: エラーハンドラーでメッセージが切り詰められました")

    @pytest.mark.parametrize("truncation_point", [100, 200, 500, 1000])
    def test_various_truncation_lengths_in_error_responses(self, truncation_point):
        """様々な長さでのメッセージ切断テスト"""
        # 異なる長さの制限を想定
        test_message = "テストメッセージ: " + "詳細情報 " * (truncation_point // 10)

        # 制限を超えるメッセージを作成
        if len(test_message) > truncation_point:
            truncated_msg = test_message[:truncation_point] + "..."

            # もし制限された長さのメッセージを返すシステムがあった場合
            received_message = truncated_msg

            # 長さが元のメッセージより短い場合、バグ
            if len(received_message) < len(test_message):
                pytest.fail(f"バグ検出: {truncation_point}文字制限でメッセージが切断されました")

    def test_exception_chain_message_lost_in_truncation(self):
        """例外チェーンでメッセージが失われる切断テスト"""
        # 複数の例外からなるチェーン
        exceptions = []
        message_parts = []

        for i in range(10):
            part = f"部分{i+1}: エラーの詳細な説明がここにあります。" * 5
            message_parts.append(part)

            try:
                raise ValueError(part)
            except ValueError as e:
                exceptions.append(e)

        # チェーンを形成
        try:
            raise RuntimeError("トップレベルのエラー") from exceptions[-1]
        except RuntimeError as chain_error:
            full_chain_msg = str(chain_error)

            # 完全なメッセージが含まれているか
            total_expected_length = sum(len(part) for part in message_parts)

            if "__cause__" in full_chain_msg or "直接原因" in full_chain_msg:
                # チェーン情報が存在するはず
                chain_info_lines = [line for line in full_chain_msg.split('\n') if "cause" in line.lower() or "原因" in line.lower()]

                if not chain_info_lines:
                    pytest.fail("バグ検出: 例外チェーン情報が失われました")

    def test_truncation_in_async_error_handling(self):
        """非同期処理でのエラーメッセージ切断テスト"""
        import asyncio
        from app.utils.error_handler import ErrorHandler

        long_async_error = "非同期エラー: " + "詳細な非同期処理の失敗メッセージ " * 50

        async def failing_async_operation():
            await asyncio.sleep(0.01)
            raise ConnectionError(long_async_error)

        async def test_async_error():
            result = await ErrorHandler.safe_execute_async(
                failing_async_operation(),
                message="Async test failed"
            )
            return result

        # 非同期実行
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                result = loop.run_until_complete(test_async_error())
            finally:
                loop.close()

            # 結果がNoneの場合（エラーハンドリングされたはず）
            if result is not None:
                pytest.fail("非同期エラーハンドリングが失敗しました")

        except HTTPException as http_exc:
            # HTTPExceptionのdetailを確認
            detail = str(http_exc.detail)

            if len(detail) < len(long_async_error) and "..." in detail:
                pytest.fail("バグ検出: 非同期エラーメッセージが切り詰められました")


class TestErrorMessageFormattingIssues:
    """エラーメッセージフォーマット問題テスト"""

    def test_message_formatting_removes_important_details(self):
        """メッセージフォーマットで重要な詳細が削除されるテスト"""
        complex_error_info = {
            "error_type": "VALIDATION_ERROR",
            "field": "strategy_parameters.slippage",
            "value": 0.05,
            "expected_range": "0.001-0.01",
            "suggestion": "スリッページ値を0.005に設定してください"
        }

        # フォーマット関数（バグのあるバージョン）
        def buggy_format_error(error_dict):
            # 制限されたフォーマット（重要な情報を削除）
            return f"エラー: {error_dict.get('error_type', 'UNKNOWN')} - 詳細はログを確認してください"

        formatted_message = buggy_format_error(complex_error_info)

        # 重要な情報が含まれていない場合、バグ
        missing_info = []
        if "slippage" not in formatted_message:
            missing_info.append("field")
        if "0.001-0.01" not in formatted_message:
            missing_info.append("expected_range")
        if "スリッページ値" not in formatted_message:
            missing_info.append("suggestion")

        if missing_info:
            pytest.fail(f"バグ検出: 重要なエラー情報が削除されました: {missing_info}")

    def test_stack_trace_gets_truncated_in_error_logs(self):
        """スタックトレースがログで切断されるテスト"""
        # 深いスタックトレースをシミュレート
        stack_trace_lines = [
            "File \"test_error_truncation.py\", line 123, in test_stack_trace_gets_truncated_in_error_logs",
            "    self.simulate_deep_call_stack()" * 50
        ] * 20

        long_stack_trace = "\n".join(stack_trace_lines)

        # ログ出力時に切断される関数
        def truncate_log_message(message, max_length=1024):
            if len(message) > max_length:
                return message[:max_length] + "\n... (truncated)"
            return message

        truncated_stack = truncate_log_message(long_stack_trace)

        if len(truncated_stack) < len(long_stack_trace):
            pytest.fail("バグ検出: スタックトレースがログで切り詰められました")

    def test_multiline_error_message_formatting_failure(self):
        """複数行エラーメッセージのフォーマット失敗テスト"""
        multiline_error = """致命的なエラー: システム障害発生
原因: データベース接続タイムアウト
影響: 取引実行が停止
解決策:
1. サーバー接続を確認
2. リトライロジックを有効化
3. ログを収集してサポートに報告"""

        # フォーマット関数が複数行を処理失敗
        def buggy_multiline_formatter(error_str):
            lines = error_str.split('\n')
            return lines[0]  # 最初の行だけ返す（バグ）

        formatted = buggy_multiline_formatter(multiline_error)

        if "解決策:" not in formatted or "データベース接続タイムアウト" not in formatted:
            pytest.fail("バグ検出: 複数行エラーメッセージの重要な情報が失われました")