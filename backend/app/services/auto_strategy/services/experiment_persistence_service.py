"""
互換ラッパー: ExperimentPersistenceService

このモジュールは移行のための互換レイヤーです。実体は
app.services.auto_strategy.persistence.experiment_persistence_service
に統合されました。将来的にはこのファイルの使用を廃止してください。
"""

from app.services.auto_strategy.persistence.experiment_persistence_service import *
