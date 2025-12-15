"""
データベース関連モジュール

アプリケーションのデータ永続化層を管理します。

主な機能:
- connection: データベース接続の管理 (SQLite/PostgreSQL)
- models: データベースモデルの定義
- migrations: Alembicによるスキーマ管理
- session: セッション管理

SQLAlchemyを使用したORMによるデータアクセスを提供し、
マイグレーションによるスキーマのバージョン管理を行います。
"""



