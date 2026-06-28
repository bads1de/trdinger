import os
import sys

# 'app' パッケージをインポートできるように、backend ディレクトリを sys.path に追加する
BACKEND_DIR = os.path.abspath(os.path.dirname(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# 相対インポートのために作業ディレクトリを BACKEND_DIR に変更する
os.chdir(BACKEND_DIR)

# カレントディレクトリからのインポートを許可する
if "." not in sys.path:
    sys.path.insert(0, ".")

# シェルがブール値以外の DEBUG 値を export していても、テスト収集を安定させる。
os.environ["DEBUG"] = "false"
