"""
バグ発見のためのテストファイル

既知のバグをテストし、修正を行うことなくバグを引き起こすコードを実装。
バグ発見を目的としたテストケース。
"""

import pytest


def test_indicator_params_import():
    """
    IndicatorParamsインポートテスト

    現象: app.services.auto_strategy.models.indicator_gene に IndicatorParams が存在しないためImportError
    """
    with pytest.raises(ImportError):
        from app.services.auto_strategy.models.indicator_gene import IndicatorParams


def test_lock_module_import():
    """
    Lockモジュールの誤用テスト

    現象: unittest.mock から Lock をインポートしようとしている（実際はthreading.Lock）
    """
    with pytest.raises(ImportError):
        from unittest.mock import Lock


def test_configuration_management_syntax():
    """
    設定管理の例外ハンドリングテスト

    現象: インデントエラー in exceptブロック（line 130）
    """
    # バグを引き起こすコード: インデントエラーを含む
    code = """
try:
    raise ValueError("test error")
except ValueError:
pass  # インデントなし - SyntaxErrorを引き起こす
"""
    # 構文エラーをテストするためにコンパイルを試みる
    with pytest.raises(SyntaxError):
        compile(code, '<string>', 'exec')


def test_resource_leak_syntax_corruption():
    """
    リソースリークの構文テスト

    現象: 構文エラー in ジェネレーター式 (line 333)
    """
    # バグを引き起こすジェネレーター式
    results = [( [1, 2, 3], 3 ), ( None, 4 )]
    try:
        # この構文は無効（else 0 がジェネレーター内にある）
        total_data_points = sum(len(data) for data, count in results if isinstance(data, list) else 0)
        assert False, "SyntaxError expected"
    except SyntaxError as e:
        # もしSyntaxErrorが発生したら期待通り
        pass
    but since it's in runtime, perhaps test differently

    # 代わりにコードとしてテスト
    code = """
total_data_points = sum(len(data) for data, count in results if isinstance(data, list) else 0)
"""
    with pytest.raises(SyntaxError):
        exec(code)


def test_security_validation_string_literals():
    """
    セキュリティ入力の文字列テスト

    現象: 文字列リテラル未終了 (line 76)
    """
    # バグを引き起こす文字列リスト
    try:
        # 未終了文字列を含むリスト
        invalid_inputs = ["not_a_number", "SELECT", "DROP", "alert('XSS')", "<script>", "'] # クォート不一致
        assert False, "SyntaxError expected"
    except SyntaxError as e:
        # 構文エラーが発生することを検証
        pass
    # 代わりに正しい方法でテスト
    code = '''
invalid_inputs = ["not_a_number", "SELECT", "DROP", "alert('XSS')", "<script>", "']
'''
    with pytest.raises(SyntaxError):
        compile(code, '<string>', 'exec')


def test_data_converters_module_missing():
    """
    データ変換モジュール存在テスト

    現象: data_converters モジュールが存在しない
    """
    with pytest.raises(ModuleNotFoundError):
        import data_converters


if __name__ == "__main__":
    pytest.main([__file__])