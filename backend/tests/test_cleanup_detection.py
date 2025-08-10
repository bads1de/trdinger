"""
pandas-ta移行後のクリーンアップ検出テスト

このテストファイルは、pandas-ta移行後に削除可能なファイルや
不要なコードを特定します。
"""

import os
import re
import sys
import ast
import pytest
from pathlib import Path
from typing import List, Dict, Set, Tuple
import importlib.util

# テスト対象のモジュールをインポート
sys.path.append(str(Path(__file__).parent.parent))


class TestCleanupDetection:
    """クリーンアップ検出テストクラス"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """テストセットアップ"""
        self.backend_path = Path(__file__).parent.parent
        self.project_root = self.backend_path.parent

    def test_detect_obsolete_files(self):
        """削除可能な古いファイルを検出"""
        obsolete_files = self._find_obsolete_files()
        
        if obsolete_files:
            print("\n削除可能な古いファイル:")
            for file_path, reason in obsolete_files:
                print(f"  {file_path}: {reason}")
        else:
            print("\n削除可能な古いファイルは見つかりませんでした。")

    def test_detect_unused_imports(self):
        """使用されていないインポートを検出"""
        unused_imports = self._find_unused_imports()
        
        if unused_imports:
            print("\n使用されていないインポート:")
            for file_path, imports in unused_imports.items():
                print(f"  {file_path}:")
                for imp in imports:
                    print(f"    - {imp}")
        else:
            print("\n使用されていないインポートは見つかりませんでした。")

    def test_detect_obsolete_aliases(self):
        """削除可能な後方互換性エイリアスを検出"""
        obsolete_aliases = self._find_obsolete_aliases()
        
        if obsolete_aliases:
            print("\n削除可能な後方互換性エイリアス:")
            for file_path, aliases in obsolete_aliases.items():
                print(f"  {file_path}:")
                for alias in aliases:
                    print(f"    - {alias}")
        else:
            print("\n削除可能なエイリアスは見つかりませんでした。")

    def test_detect_duplicate_implementations(self):
        """重複する実装を検出"""
        duplicates = self._find_duplicate_implementations()
        
        if duplicates:
            print("\n重複する実装:")
            for func_name, locations in duplicates.items():
                if len(locations) > 1:
                    print(f"  {func_name}:")
                    for location in locations:
                        print(f"    - {location}")
        else:
            print("\n重複する実装は見つかりませんでした。")

    def test_detect_obsolete_documentation(self):
        """削除可能な古いドキュメントを検出"""
        obsolete_docs = self._find_obsolete_documentation()
        
        if obsolete_docs:
            print("\n削除可能な古いドキュメント:")
            for doc_path, reason in obsolete_docs:
                print(f"  {doc_path}: {reason}")
        else:
            print("\n削除可能な古いドキュメントは見つかりませんでした。")

    def test_detect_unused_test_files(self):
        """使用されていないテストファイルを検出"""
        unused_tests = self._find_unused_test_files()
        
        if unused_tests:
            print("\n削除可能なテストファイル:")
            for test_path, reason in unused_tests:
                print(f"  {test_path}: {reason}")
        else:
            print("\n削除可能なテストファイルは見つかりませんでした。")

    def _find_obsolete_files(self) -> List[Tuple[str, str]]:
        """削除可能な古いファイルを検索"""
        obsolete_files = []
        
        # 検索対象のパターン
        obsolete_patterns = [
            (r".*talib.*adapter.*\.py$", "talib adapter files"),
            (r".*talib.*wrapper.*\.py$", "talib wrapper files"),
            (r".*talib.*migration.*\.py$", "migration temporary files"),
            (r".*\.bak$", "backup files"),
            (r".*\.old$", "old files"),
            (r".*\.tmp$", "temporary files"),
        ]
        
        for py_file in self._get_all_files():
            file_str = str(py_file)
            for pattern, reason in obsolete_patterns:
                if re.match(pattern, file_str, re.IGNORECASE):
                    obsolete_files.append((file_str, reason))
                    break
        
        return obsolete_files

    def _find_unused_imports(self) -> Dict[str, List[str]]:
        """使用されていないインポートを検索"""
        unused_imports = {}
        
        for py_file in self._get_python_files():
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # ASTを使用してインポートと使用を分析
                tree = ast.parse(content)
                imports = self._extract_imports(tree)
                used_names = self._extract_used_names(tree)
                
                unused = []
                for imp_name, imp_line in imports:
                    if imp_name not in used_names:
                        unused.append(f"Line {imp_line}: {imp_name}")
                
                if unused:
                    unused_imports[str(py_file)] = unused
                    
            except Exception:
                continue
        
        return unused_imports

    def _find_obsolete_aliases(self) -> Dict[str, List[str]]:
        """削除可能な後方互換性エイリアスを検索"""
        obsolete_aliases = {}
        
        # pandas_ta_utils.pyの後方互換性エイリアスをチェック
        utils_file = self.backend_path / 'app' / 'services' / 'indicators' / 'pandas_ta_utils.py'
        
        if utils_file.exists():
            try:
                with open(utils_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 後方互換性エイリアスのパターンを検索
                alias_pattern = r'pandas_ta_(\w+)\s*=\s*(\w+)'
                matches = re.findall(alias_pattern, content)
                
                if matches:
                    aliases = []
                    for old_name, new_name in matches:
                        # エイリアスが実際に使用されているかチェック
                        if not self._is_alias_used(f"pandas_ta_{old_name}"):
                            aliases.append(f"pandas_ta_{old_name} = {new_name}")
                    
                    if aliases:
                        obsolete_aliases[str(utils_file)] = aliases
                        
            except Exception:
                pass
        
        return obsolete_aliases

    def _find_duplicate_implementations(self) -> Dict[str, List[str]]:
        """重複する実装を検索"""
        function_locations = {}
        
        for py_file in self._get_python_files():
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_name = node.name
                        if func_name not in function_locations:
                            function_locations[func_name] = []
                        function_locations[func_name].append(str(py_file))
                        
            except Exception:
                continue
        
        # 重複のみを返す
        duplicates = {name: locations for name, locations in function_locations.items() 
                     if len(locations) > 1}
        
        return duplicates

    def _find_obsolete_documentation(self) -> List[Tuple[str, str]]:
        """削除可能な古いドキュメントを検索"""
        obsolete_docs = []
        
        docs_patterns = [
            (r".*talib.*usage.*\.md$", "talib usage documentation"),
            (r".*talib.*inventory.*\.md$", "talib inventory documentation"),
            (r".*migration.*plan.*\.md$", "migration plan documentation"),
            (r".*migration.*progress.*\.md$", "migration progress documentation"),
        ]
        
        docs_dir = self.backend_path / 'docs'
        if docs_dir.exists():
            for doc_file in docs_dir.rglob('*.md'):
                doc_str = str(doc_file)
                for pattern, reason in docs_patterns:
                    if re.match(pattern, doc_str, re.IGNORECASE):
                        # ドキュメントが参照目的でない場合は削除候補
                        if not self._is_reference_documentation(doc_file):
                            obsolete_docs.append((doc_str, reason))
                        break
        
        return obsolete_docs

    def _find_unused_test_files(self) -> List[Tuple[str, str]]:
        """使用されていないテストファイルを検索"""
        unused_tests = []
        
        test_patterns = [
            (r".*test.*talib.*migration.*\.py$", "migration test files"),
            (r".*test.*talib.*adapter.*\.py$", "adapter test files"),
        ]
        
        tests_dir = self.backend_path / 'tests'
        if tests_dir.exists():
            for test_file in tests_dir.rglob('test_*.py'):
                test_str = str(test_file)
                for pattern, reason in test_patterns:
                    if re.match(pattern, test_str, re.IGNORECASE):
                        # テストが一時的なものかチェック
                        if self._is_temporary_test_file(test_file):
                            unused_tests.append((test_str, reason))
                        break
        
        return unused_tests

    def _extract_imports(self, tree: ast.AST) -> List[Tuple[str, int]]:
        """ASTからインポート文を抽出"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((alias.name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports.append((alias.name, node.lineno))
        
        return imports

    def _extract_used_names(self, tree: ast.AST) -> Set[str]:
        """ASTから使用されている名前を抽出"""
        used_names = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
        
        return used_names

    def _is_alias_used(self, alias_name: str) -> bool:
        """エイリアスが使用されているかチェック"""
        for py_file in self._get_python_files():
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if alias_name in content:
                    return True
                    
            except Exception:
                continue
        
        return False

    def _is_reference_documentation(self, doc_file: Path) -> bool:
        """参照用ドキュメントかチェック"""
        try:
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 参照用キーワードをチェック
            reference_keywords = [
                "参照用",
                "reference",
                "mapping",
                "対応表",
                "リファレンス"
            ]
            
            content_lower = content.lower()
            return any(keyword in content_lower for keyword in reference_keywords)
            
        except Exception:
            return False

    def _is_temporary_test_file(self, test_file: Path) -> bool:
        """一時的なテストファイルかチェック"""
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 一時的なテストのキーワードをチェック
            temporary_keywords = [
                "一時的",
                "temporary",
                "migration",
                "移行用",
                "削除予定"
            ]
            
            content_lower = content.lower()
            return any(keyword in content_lower for keyword in temporary_keywords)
            
        except Exception:
            return False

    def _get_all_files(self) -> List[Path]:
        """全ファイルのリストを取得"""
        all_files = []
        
        # backend ディレクトリ内の全ファイルを検索
        for file_path in self.backend_path.rglob('*'):
            if file_path.is_file():
                all_files.append(file_path)
        
        return all_files

    def _get_python_files(self) -> List[Path]:
        """Python ファイルのリストを取得"""
        python_files = []
        
        # backend ディレクトリ内のPythonファイルを検索
        for py_file in self.backend_path.rglob('*.py'):
            # __pycache__ ディレクトリは除外
            if '__pycache__' not in str(py_file):
                python_files.append(py_file)
        
        return python_files


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
