"""
メソッドミスマッチ検証テスト

このテストスイートは、サービスのメソッド存在確認で発生する問題を検出します。
正しいメソッド名にもかかわらず存在しないと判断されるケースや、
存在しないメソッドが呼び出されるケースをテストします。
"""

import pytest
from unittest.mock import MagicMock, patch, Mock
import inspect


class TestMethodMismatchValidation:
    """メソッドミスマッチバグ検出テスト"""

    def test_service_method_not_found_false_positive(self):
        """存在するメソッドが「見つからない」と誤判定されるバグテスト"""
        from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService

        # 実際のメソッドをチェック
        service = AutoStrategyService()
        real_methods = [method for method in dir(service) if not method.startswith('_')]

        # 存在するメソッドをランダムにチェック
        test_method = "start_strategy_generation" if hasattr(service, "start_strategy_generation") else real_methods[0]

        try:
            # メソッド存在チェック関数（バグのあるバージョン）
            def buggy_method_exists(obj, method_name):
                # バグ: hasattrを使用せず、直接dir()結果をチェック
                return method_name in [m.upper() for m in dir(obj)]  # 大文字変換でミスマッチ

            # 存在するメソッドが「ない」と判定される（バグ）
            exists = buggy_method_exists(service, test_method)

            if not exists:
                pytest.fail(f"バグ検出: 存在するメソッド '{test_method}' が未検出と判定されました")

        except AttributeError:
            pytest.fail("バグ検出: メソッド存在チェック関数自体が不具合")

    def test_nonexistent_method_called_without_validation(self):
        """存在しないメソッドが呼び出される前の検証失敗テスト"""

        class MockService:
            def existing_method(self):
                return "exists"

            def __getattr__(self, name):
                # 存在しないメソッドが呼ばれたらエラーを出すが、バグで出さない場合
                if name.startswith('nonexistent_'):
                    return lambda *args, **kwargs: "メソッドが存在しないのに呼び出されました"
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        service = MockService()

        # 存在しないメソッドを呼ぼうとする
        nonexistent_method_name = "nonexistent_method_123"

        try:
            # 事前検証なしでメソッド呼び出し
            method_ref = getattr(service, nonexistent_method_name, None)

            if method_ref is not None:
                result = method_ref()
                # バグ: 存在しないメソッドが呼び出せてしまう
                if "存在しないのに呼び出されました" in result:
                    pytest.fail(f"バグ検出: 存在しないメソッド '{nonexistent_method_name}' が呼び出せました")

        except AttributeError:
            # これが正しい動作
            assert True

    def test_dynamic_method_creation_fails_validation_check(self):
        """動的メソッド作成が検証チェックを回避するテスト"""

        class DynamicService:
            def __init__(self):
                self._dynamic_methods = {}

            def add_dynamic_method(self, name, func):
                self._dynamic_methods[name] = func

            def __getattr__(self, name):
                if name in self._dynamic_methods:
                    return self._dynamic_methods[name]
                raise AttributeError(f"DyanmicService has no method '{name}'")

            def validate_method_exists(self, method_name):
                # バグ: 動的メソッドを考慮しないチェック
                return hasattr(self, method_name) and not callable(getattr(self, method_name))

        service = DynamicService()

        # 動的メソッドを追加
        service.add_dynamic_method("dynamic_test_method", lambda: "動的メソッド実行結果")

        # 検証関数でチェック
        exists_via_validation = service.validate_method_exists("dynamic_test_method")

        if not exists_via_validation:
            pytest.fail("バグ検出: 動的作成メソッドが検証関数で検出されませんでした")

    def test_method_signature_mismatch_not_detected(self):
        """メソッドシグネチャミスマッチが検出されないバグテスト"""

        class StrictService:
            def process_data(self, data: dict, config: dict, validate=True):
                """期待されるシグネチャ"""
                return f"処理成功: {len(data)}件"

        service = StrictService()

        # 別シグネチャの関数
        def wrong_signature_func(data_only):
            return f"間違った処理: {data_only}"

        # メソッド検証（パラメータチェック）
        def validate_method_signature(obj, method_name, expected_params):
            if not hasattr(obj, method_name):
                return False

            method = getattr(obj, method_name)
            if not callable(method):
                return False

            sig = inspect.signature(method)
            param_names = list(sig.parameters.keys())

            # バグ: パラメータチェックが不十分
            return len(param_names) >= len(expected_params)  # 一部だけチェック

        # 正しいパラメータ
        correct_exists = validate_method_signature(service, "process_data", ["data", "config", "validate"])

        # 間違った関数のパラメータ
        setattr(service, "wrong_process", wrong_signature_func)
        wrong_exists = validate_method_signature(service, "wrong_process", ["data", "config", "validate"])

        if wrong_exists == correct_exists:
            pytest.fail("バグ検出: シグネチャミスマッチが検出されませんでした")

    def test_import_error_hides_real_method_availability(self):
        """インポートエラーが本来のメソッド可用性を隠すテスト"""

        # サービスが依存するモジュールのインポートに失敗するケース
        mock_service_import_error = """

from some.nonexistent.module import RequiredClass

class ServiceWithImportError:
    def __init__(self):
        try:
            self.required_obj = RequiredClass()
        except ImportError:
            # バグ: インポートエラーを無視して空のオブジェクトを設定
            self.required_obj = None

    def use_required_method(self):
        if self.required_obj:
            return self.required_obj.some_method()
        else:
            # バグ: Noneオブジェクトのメソッドを呼び出す
            return self.required_obj.some_method()  # AttributeErrorになるはず

    def check_method_available(self, method_name):
        # バグ: インポートエラーの影響を考慮しないチェック
        return hasattr(self, method_name)
"""

        # このようなサービスクラスをテスト
        try:
            exec(mock_service_import_error)
            service = eval("ServiceWithImportError()")

            # メソッド存在チェック
            available = service.check_method_available("use_required_method")

            if available:
                try:
                    result = service.use_required_method()
                    pytest.fail("バグ検出: インポートエラーがメソッド可用性を誤って報告しました")
                except AttributeError:
                    # これは想定される
                    assert True
        except Exception as e:
            # エラー自体が問題
            pytest.fail(f"バグ検出: インポートエラーハンドリングが不適切 {e}")

    def test_method_cache_invalidation_bug(self):
        """メソッドキャッシュ無効化バグテスト"""

        class CachedService:
            def __init__(self):
                self._method_cache = {}

            def _cache_method_presence(self, method_name, exists):
                self._method_cache[method_name] = exists

            def check_method_cached(self, method_name):
                if method_name in self._method_cache:
                    return self._method_cache[method_name]

                # 実チェック
                exists = hasattr(self, method_name)
                self._cache_method_presence(method_name, exists)
                return exists

            def real_method(self):
                return "実メソッド存在"

        service = CachedService()

        # 最初はキャッシュなしでチェック
        first_check = service.check_method_cached("real_method")
        assert first_check == True

        # メソッドを削除してみる（シミュレーション）
        original_method = service.real_method
        delattr(service, "real_method")

        # キャッシュが無効化されないバグ
        second_check = service.check_method_cached("real_method")
        if second_check == True:
            pytest.fail("バグ検出: メソッド削除後もキャッシュが無効化されませんでした")

        # キャッシュクリア
        service._method_cache.clear()
        third_check = service.check_method_cached("real_method")
        assert third_check == False

    def test_method_name_case_sensitivity_bug(self):
        """メソッド名の大文字小文字区別バグテスト"""

        class CaseSensitiveService:
            def TestMethod(self):
                return "テストメソッド"

            def validate_method_name(self, name):
                # バグ: 大文字小文字を区別したチェックしかしない
                return hasattr(self, name)

        service = CaseSensitiveService()

        # 存在するメソッドを間違ったケースでチェック
        check_upper = service.validate_method_name("testmethod")  # 小文字
        if check_upper == False:
            pytest.fail("バグ検出: メソッド名の大文字小文字が厳しくチェックされすぎます")


class TestServiceMethodDiscoveryIssues:
    """サービスメソッド発見問題テスト"""

    def test_method_discovery_excludes_valid_methods(self):
        """有効なメソッドをメソッド発見から除外するテスト"""

        class MethodRichService:
            def public_method_1(self): pass
            def public_method_2(self): pass

            def _private_method_1(self): pass
            def _private_method_2(self): pass

            @property
            def some_property(self): return "prop"

            @classmethod
            def class_method(cls): pass

            @staticmethod
            def static_method(): pass

        service = MethodRichService()

        def discover_methods_badly(obj):
            """バグのあるメソッド発見関数"""
            all_attrs = dir(obj)
            # バグ: クラスメソッドとスタティックメソッドを除外しすぎる
            methods = []
            for attr in all_attrs:
                if not attr.startswith('_'):
                    try:
                        val = getattr(obj, attr)
                        if callable(val) and not isinstance(val, property):
                            # バグ: クラスメソッドを除外
                            if not hasattr(val, '__self__') or val.__self__ == obj:
                                methods.append(attr)
                    except:
                        pass
            return methods

        discovered_methods = discover_methods_badly(service)

        # クラスメソッドとスタティックメソッドが含まれているべき
        expected_methods = ['public_method_1', 'public_method_2', 'class_method', 'static_method']

        missing_methods = set(expected_methods) - set(discovered_methods)

        if missing_methods:
            pytest.fail(f"バグ検出: 有効なメソッドが発見から除外されました: {missing_methods}")

    def test_inheritance_breaks_method_discovery(self):
        """継承関係がメソッド発見を破壊するテスト"""

        class BaseService:
            def base_method(self): return "base"

        class DerivedService(BaseService):
            def derived_method(self): return "derived"

        derived_service = DerivedService()

        def discover_methods_with_inheritance_bug(obj):
            """バグのある継承考慮メソッド発見"""
            # バグ: 継承メソッドを正しく発見しない
            return [m for m in dir(obj) if not m.startswith('_') and callable(getattr(obj, m, None))]

        discovered = discover_methods_with_inheritance_bug(derived_service)

        if "base_method" not in discovered:
            pytest.fail("バグ検出: 継承されたメソッドが発見されませんでした")

    def test_method_discovery_with_decorators_fails(self):
        """デコレータ付きメソッドの発見失敗テスト"""

        def some_decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

        class DecoratedService:
            @some_decorator
            def decorated_method(self):
                return "decorated"

            @property
            @some_decorator  # これは問題になる可能性
            def decorated_property(self):
                return "prop"

        service = DecoratedService()

        def discover_decorated_methods_badly(obj):
            """バグのあるデコレータ考慮メソッド発見"""
            methods = []
            for attr_name in dir(obj):
                if not attr_name.startswith('_'):
                    attr = getattr(obj, attr_name)
                    if callable(attr):
                        # バグ: デコレータの__wrapped__を考慮しない
                        orig_func = getattr(attr, '__wrapped__', None)
                        if orig_func:
                            methods.append(attr_name)
                        elif not hasattr(attr, '__wrapped__'):
                            methods.append(attr_name)
            return methods

        discovered = discover_decorated_methods_badly(service)

        if "decorated_method" not in discovered:
            pytest.fail("バグ検出: デコレータ付きメソッドが発見されませんでした")