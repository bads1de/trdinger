"""
オートストラテジー データ整合性・一貫性テスト

データソース間の整合性、バックアップ・復元、分散処理、ACID特性を検証します。
"""

import sys
import os

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, backend_dir)

import pytest
import pandas as pd
import numpy as np
import time
import sqlite3
import tempfile
import shutil
import threading
import concurrent.futures
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
import hashlib

logger = logging.getLogger(__name__)


class TestDataConsistency:
    """データ整合性・一貫性テストクラス"""
    
    def setup_method(self):
        """テスト前の準備"""
        self.start_time = time.time()
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """テスト後のクリーンアップ"""
        execution_time = time.time() - self.start_time
        logger.info(f"テスト実行時間: {execution_time:.3f}秒")
        
        # 一時ディレクトリのクリーンアップ
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
    
    def create_test_database(self, db_path: str) -> None:
        """テスト用データベースを作成"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # テーブル作成
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open_price REAL NOT NULL,
                high_price REAL NOT NULL,
                low_price REAL NOT NULL,
                close_price REAL NOT NULL,
                volume REAL NOT NULL,
                checksum TEXT,
                created_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                sl_price REAL,
                tp_price REAL,
                risk_reward_ratio REAL,
                checksum TEXT,
                created_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """)
        
        conn.commit()
        conn.close()
    
    def calculate_data_checksum(self, data: Dict) -> str:
        """データのチェックサムを計算"""
        # 辞書を文字列に変換してハッシュ化
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def test_multi_source_data_consistency(self):
        """テスト53: 異なるデータソース間での価格データ整合性"""
        logger.info("🔍 マルチソースデータ整合性テスト開始")
        
        try:
            # 複数のデータソースをシミュレート
            data_sources = {
                "source_a": [],
                "source_b": [],
                "source_c": []
            }
            
            # 基準データを生成
            base_time = datetime(2023, 1, 1, 0, 0, 0)
            num_points = 1000
            
            for i in range(num_points):
                timestamp = base_time + timedelta(minutes=i)
                base_price = 50000 + np.random.normal(0, 100)
                
                # 各ソースで微小な差異を持つデータを生成
                for source_name in data_sources.keys():
                    # ソース間で最大0.1%の価格差を許容
                    price_variance = np.random.normal(0, base_price * 0.001)
                    
                    data_point = {
                        "timestamp": timestamp.isoformat(),
                        "symbol": "BTC/USDT",
                        "open": base_price + price_variance,
                        "high": base_price + price_variance + abs(np.random.normal(0, 50)),
                        "low": base_price + price_variance - abs(np.random.normal(0, 50)),
                        "close": base_price + price_variance + np.random.normal(0, 20),
                        "volume": np.random.exponential(1000),
                        "source": source_name
                    }
                    
                    data_sources[source_name].append(data_point)
            
            # データ整合性の分析
            consistency_stats = {
                "price_differences": [],
                "volume_differences": [],
                "timestamp_mismatches": 0,
                "outliers": 0
            }
            
            # 同一時刻のデータを比較
            for i in range(num_points):
                source_a_data = data_sources["source_a"][i]
                source_b_data = data_sources["source_b"][i]
                source_c_data = data_sources["source_c"][i]
                
                # タイムスタンプの一致確認
                if not (source_a_data["timestamp"] == source_b_data["timestamp"] == source_c_data["timestamp"]):
                    consistency_stats["timestamp_mismatches"] += 1
                
                # 価格差の分析
                prices_a = [source_a_data["open"], source_a_data["high"], source_a_data["low"], source_a_data["close"]]
                prices_b = [source_b_data["open"], source_b_data["high"], source_b_data["low"], source_b_data["close"]]
                prices_c = [source_c_data["open"], source_c_data["high"], source_c_data["low"], source_c_data["close"]]
                
                for j in range(4):  # OHLC
                    max_price = max(prices_a[j], prices_b[j], prices_c[j])
                    min_price = min(prices_a[j], prices_b[j], prices_c[j])
                    price_diff_pct = (max_price - min_price) / min_price * 100
                    
                    consistency_stats["price_differences"].append(price_diff_pct)
                    
                    # 異常な価格差（1%以上）を検出
                    if price_diff_pct > 1.0:
                        consistency_stats["outliers"] += 1
                
                # ボリューム差の分析
                volumes = [source_a_data["volume"], source_b_data["volume"], source_c_data["volume"]]
                max_volume = max(volumes)
                min_volume = min(volumes)
                if min_volume > 0:
                    volume_diff_pct = (max_volume - min_volume) / min_volume * 100
                    consistency_stats["volume_differences"].append(volume_diff_pct)
            
            # 整合性統計の分析
            avg_price_diff = np.mean(consistency_stats["price_differences"])
            max_price_diff = max(consistency_stats["price_differences"])
            p95_price_diff = np.percentile(consistency_stats["price_differences"], 95)
            
            avg_volume_diff = np.mean(consistency_stats["volume_differences"])
            max_volume_diff = max(consistency_stats["volume_differences"])
            
            logger.info(f"マルチソースデータ整合性結果:")
            logger.info(f"  データポイント数: {num_points}")
            logger.info(f"  平均価格差: {avg_price_diff:.4f}%")
            logger.info(f"  最大価格差: {max_price_diff:.4f}%")
            logger.info(f"  95%ile価格差: {p95_price_diff:.4f}%")
            logger.info(f"  平均ボリューム差: {avg_volume_diff:.2f}%")
            logger.info(f"  最大ボリューム差: {max_volume_diff:.2f}%")
            logger.info(f"  タイムスタンプ不一致: {consistency_stats['timestamp_mismatches']}")
            logger.info(f"  価格異常値: {consistency_stats['outliers']}")
            
            # 整合性要件の確認
            assert avg_price_diff < 0.5, f"平均価格差が大きすぎます: {avg_price_diff:.4f}%"
            assert max_price_diff < 2.0, f"最大価格差が大きすぎます: {max_price_diff:.4f}%"
            assert consistency_stats["timestamp_mismatches"] == 0, f"タイムスタンプ不一致があります: {consistency_stats['timestamp_mismatches']}"
            assert consistency_stats["outliers"] < num_points * 0.01, f"異常値が多すぎます: {consistency_stats['outliers']}"
            
            logger.info("✅ マルチソースデータ整合性テスト成功")
            
        except Exception as e:
            pytest.fail(f"マルチソースデータ整合性テストエラー: {e}")
    
    def test_backup_restore_consistency(self):
        """テスト54: バックアップ・復元後のデータ一貫性"""
        logger.info("🔍 バックアップ・復元一貫性テスト開始")
        
        try:
            # 元のデータベースを作成
            original_db = os.path.join(self.temp_dir, "original.db")
            backup_db = os.path.join(self.temp_dir, "backup.db")
            restored_db = os.path.join(self.temp_dir, "restored.db")
            
            self.create_test_database(original_db)
            
            # テストデータを挿入
            conn = sqlite3.connect(original_db)
            cursor = conn.cursor()
            
            test_data = []
            for i in range(1000):
                timestamp = int(time.time()) - (1000 - i) * 60  # 1分間隔
                data = {
                    "symbol": "BTC/USDT",
                    "timestamp": timestamp,
                    "open_price": 50000 + np.random.normal(0, 100),
                    "high_price": 50000 + np.random.normal(0, 100) + 50,
                    "low_price": 50000 + np.random.normal(0, 100) - 50,
                    "close_price": 50000 + np.random.normal(0, 100),
                    "volume": np.random.exponential(1000)
                }
                
                # チェックサムを計算
                checksum = self.calculate_data_checksum(data)
                data["checksum"] = checksum
                
                cursor.execute("""
                    INSERT INTO market_data (symbol, timestamp, open_price, high_price, low_price, close_price, volume, checksum)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (data["symbol"], data["timestamp"], data["open_price"], data["high_price"], 
                     data["low_price"], data["close_price"], data["volume"], data["checksum"]))
                
                test_data.append(data)
            
            conn.commit()
            
            # 元データの統計を取得
            cursor.execute("SELECT COUNT(*), SUM(volume), AVG(close_price) FROM market_data")
            original_stats = cursor.fetchone()
            
            cursor.execute("SELECT checksum FROM market_data ORDER BY id")
            original_checksums = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            # バックアップ作成
            logger.info("バックアップ作成中...")
            shutil.copy2(original_db, backup_db)
            
            # バックアップファイルの整合性確認
            assert os.path.exists(backup_db), "バックアップファイルが作成されていません"
            
            backup_size = os.path.getsize(backup_db)
            original_size = os.path.getsize(original_db)
            assert backup_size == original_size, f"バックアップサイズが異なります: {backup_size} vs {original_size}"
            
            # 復元処理
            logger.info("復元処理中...")
            shutil.copy2(backup_db, restored_db)
            
            # 復元データの検証
            conn = sqlite3.connect(restored_db)
            cursor = conn.cursor()
            
            # 統計の比較
            cursor.execute("SELECT COUNT(*), SUM(volume), AVG(close_price) FROM market_data")
            restored_stats = cursor.fetchone()
            
            assert restored_stats == original_stats, f"統計データが一致しません: {restored_stats} vs {original_stats}"
            
            # チェックサムの比較
            cursor.execute("SELECT checksum FROM market_data ORDER BY id")
            restored_checksums = [row[0] for row in cursor.fetchall()]
            
            assert restored_checksums == original_checksums, "チェックサムが一致しません"
            
            # 個別レコードの検証
            cursor.execute("SELECT * FROM market_data ORDER BY id")
            restored_records = cursor.fetchall()
            
            conn.close()
            
            # データ整合性の詳細検証
            integrity_check_passed = 0
            for i, record in enumerate(restored_records):
                record_data = {
                    "symbol": record[1],
                    "timestamp": record[2],
                    "open_price": record[3],
                    "high_price": record[4],
                    "low_price": record[5],
                    "close_price": record[6],
                    "volume": record[7]
                }
                
                expected_checksum = self.calculate_data_checksum(record_data)
                actual_checksum = record[8]
                
                if expected_checksum == actual_checksum:
                    integrity_check_passed += 1
                else:
                    logger.warning(f"レコード {i+1}: チェックサム不一致")
            
            integrity_rate = integrity_check_passed / len(restored_records)
            
            logger.info(f"バックアップ・復元結果:")
            logger.info(f"  元データ件数: {original_stats[0]}")
            logger.info(f"  復元データ件数: {restored_stats[0]}")
            logger.info(f"  整合性チェック通過率: {integrity_rate:.1%}")
            logger.info(f"  ファイルサイズ一致: {backup_size == original_size}")
            
            # 整合性要件の確認
            assert integrity_rate == 1.0, f"データ整合性が完全ではありません: {integrity_rate:.1%}"
            assert restored_stats[0] == original_stats[0], "レコード数が一致しません"
            
            logger.info("✅ バックアップ・復元一貫性テスト成功")
            
        except Exception as e:
            pytest.fail(f"バックアップ・復元一貫性テストエラー: {e}")
    
    def test_distributed_processing_synchronization(self):
        """テスト55: 分散処理環境でのデータ同期"""
        logger.info("🔍 分散処理データ同期テスト開始")
        
        try:
            from app.services.auto_strategy.calculators.tpsl_calculator import TPSLCalculator
            
            # 分散処理ノードをシミュレート
            num_nodes = 4
            shared_data = {
                "processed_items": 0,
                "results": [],
                "lock": threading.Lock(),
                "errors": []
            }
            
            # 処理対象データ
            input_data = []
            for i in range(1000):
                data_item = {
                    "id": i,
                    "price": 50000 + np.random.normal(0, 100),
                    "sl_pct": 0.02,
                    "tp_pct": 0.04,
                    "direction": 1.0 if i % 2 == 0 else -1.0
                }
                input_data.append(data_item)
            
            def process_node(node_id: int, data_chunk: List[Dict]) -> Dict:
                """分散処理ノード"""
                calculator = TPSLCalculator()
                node_results = []
                node_errors = []
                
                for item in data_chunk:
                    try:
                        # TP/SL計算
                        sl_price, tp_price = calculator.calculate_basic_tpsl_prices(
                            item["price"], item["sl_pct"], item["tp_pct"], item["direction"]
                        )
                        
                        result = {
                            "id": item["id"],
                            "node_id": node_id,
                            "sl_price": sl_price,
                            "tp_price": tp_price,
                            "processed_at": time.time()
                        }
                        
                        node_results.append(result)
                        
                        # 共有データの更新（同期）
                        with shared_data["lock"]:
                            shared_data["processed_items"] += 1
                            shared_data["results"].append(result)
                        
                    except Exception as e:
                        node_errors.append({"id": item["id"], "error": str(e)})
                        with shared_data["lock"]:
                            shared_data["errors"].append({"node_id": node_id, "item_id": item["id"], "error": str(e)})
                
                return {
                    "node_id": node_id,
                    "processed_count": len(node_results),
                    "error_count": len(node_errors),
                    "results": node_results
                }
            
            # データを分散処理ノードに分割
            chunk_size = len(input_data) // num_nodes
            data_chunks = [
                input_data[i:i + chunk_size] 
                for i in range(0, len(input_data), chunk_size)
            ]
            
            # 分散処理実行
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_nodes) as executor:
                futures = [
                    executor.submit(process_node, i, chunk) 
                    for i, chunk in enumerate(data_chunks)
                ]
                
                node_results = [future.result() for future in futures]
            
            processing_time = time.time() - start_time
            
            # 同期結果の分析
            total_processed = sum(result["processed_count"] for result in node_results)
            total_errors = sum(result["error_count"] for result in node_results)
            
            # データ重複チェック
            all_ids = [result["id"] for result in shared_data["results"]]
            unique_ids = set(all_ids)
            duplicate_count = len(all_ids) - len(unique_ids)
            
            # データ欠損チェック
            expected_ids = set(range(len(input_data)))
            processed_ids = set(all_ids)
            missing_ids = expected_ids - processed_ids
            
            logger.info(f"分散処理同期結果:")
            logger.info(f"  処理ノード数: {num_nodes}")
            logger.info(f"  総処理件数: {total_processed}")
            logger.info(f"  エラー件数: {total_errors}")
            logger.info(f"  処理時間: {processing_time:.3f}秒")
            logger.info(f"  スループット: {total_processed/processing_time:.1f}件/秒")
            logger.info(f"  重複データ: {duplicate_count}件")
            logger.info(f"  欠損データ: {len(missing_ids)}件")
            
            # ノード別統計
            for result in node_results:
                logger.info(f"  ノード{result['node_id']}: {result['processed_count']}件処理, {result['error_count']}件エラー")
            
            # 同期要件の確認
            assert duplicate_count == 0, f"データ重複が発生しました: {duplicate_count}件"
            assert len(missing_ids) == 0, f"データ欠損が発生しました: {len(missing_ids)}件"
            assert total_processed == len(input_data), f"処理件数が一致しません: {total_processed} vs {len(input_data)}"
            assert total_errors < len(input_data) * 0.01, f"エラー率が高すぎます: {total_errors}/{len(input_data)}"
            
            # 結果の一貫性確認
            for result in shared_data["results"][:10]:  # 最初の10件をチェック
                assert result["sl_price"] is not None, "SL価格がNullです"
                assert result["tp_price"] is not None, "TP価格がNullです"
                assert result["sl_price"] > 0, "SL価格が負です"
                assert result["tp_price"] > 0, "TP価格が負です"
            
            logger.info("✅ 分散処理データ同期テスト成功")

        except Exception as e:
            pytest.fail(f"分散処理データ同期テストエラー: {e}")

    def test_transaction_atomicity(self):
        """テスト56: トランザクション処理の原子性（ACID特性）"""
        logger.info("🔍 トランザクション原子性テスト開始")

        try:
            # テスト用データベース
            test_db = os.path.join(self.temp_dir, "transaction_test.db")
            self.create_test_database(test_db)

            # トランザクション統計
            transaction_stats = {
                "successful_transactions": 0,
                "failed_transactions": 0,
                "rollback_count": 0,
                "data_consistency_errors": 0
            }

            def execute_transaction(transaction_id: int, should_fail: bool = False) -> bool:
                """トランザクションを実行"""
                conn = sqlite3.connect(test_db)
                cursor = conn.cursor()

                try:
                    # トランザクション開始
                    cursor.execute("BEGIN TRANSACTION")

                    # 複数の関連操作を実行
                    timestamp = int(time.time()) + transaction_id

                    # 市場データ挿入
                    market_data = {
                        "symbol": "BTC/USDT",
                        "timestamp": timestamp,
                        "open_price": 50000 + np.random.normal(0, 100),
                        "high_price": 50000 + np.random.normal(0, 100) + 50,
                        "low_price": 50000 + np.random.normal(0, 100) - 50,
                        "close_price": 50000 + np.random.normal(0, 100),
                        "volume": np.random.exponential(1000)
                    }

                    checksum = self.calculate_data_checksum(market_data)

                    cursor.execute("""
                        INSERT INTO market_data (symbol, timestamp, open_price, high_price, low_price, close_price, volume, checksum)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (market_data["symbol"], market_data["timestamp"], market_data["open_price"],
                         market_data["high_price"], market_data["low_price"], market_data["close_price"],
                         market_data["volume"], checksum))

                    market_data_id = cursor.lastrowid

                    # 戦略結果挿入
                    strategy_result = {
                        "strategy_name": f"test_strategy_{transaction_id}",
                        "symbol": "BTC/USDT",
                        "timestamp": timestamp,
                        "sl_price": market_data["close_price"] * 0.98,
                        "tp_price": market_data["close_price"] * 1.04,
                        "risk_reward_ratio": 2.0
                    }

                    strategy_checksum = self.calculate_data_checksum(strategy_result)

                    cursor.execute("""
                        INSERT INTO strategy_results (strategy_name, symbol, timestamp, sl_price, tp_price, risk_reward_ratio, checksum)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (strategy_result["strategy_name"], strategy_result["symbol"], strategy_result["timestamp"],
                         strategy_result["sl_price"], strategy_result["tp_price"], strategy_result["risk_reward_ratio"],
                         strategy_checksum))

                    # 意図的な失敗をシミュレート
                    if should_fail:
                        raise Exception(f"Intentional failure for transaction {transaction_id}")

                    # トランザクションコミット
                    conn.commit()
                    transaction_stats["successful_transactions"] += 1
                    return True

                except Exception as e:
                    # ロールバック
                    conn.rollback()
                    transaction_stats["failed_transactions"] += 1
                    transaction_stats["rollback_count"] += 1
                    logger.debug(f"Transaction {transaction_id} rolled back: {e}")
                    return False

                finally:
                    conn.close()

            # 複数のトランザクションを並行実行
            num_transactions = 100
            failure_rate = 0.1  # 10%の失敗率

            start_time = time.time()

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for i in range(num_transactions):
                    should_fail = np.random.random() < failure_rate
                    future = executor.submit(execute_transaction, i, should_fail)
                    futures.append(future)

                results = [future.result() for future in futures]

            processing_time = time.time() - start_time

            # データ整合性の検証
            conn = sqlite3.connect(test_db)
            cursor = conn.cursor()

            # 市場データとストラテジー結果の件数確認
            cursor.execute("SELECT COUNT(*) FROM market_data")
            market_data_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM strategy_results")
            strategy_results_count = cursor.fetchone()[0]

            # 成功したトランザクションと実際のデータ件数が一致することを確認
            expected_count = transaction_stats["successful_transactions"]

            # チェックサムの整合性確認
            cursor.execute("SELECT * FROM market_data")
            market_records = cursor.fetchall()

            checksum_errors = 0
            for record in market_records:
                record_data = {
                    "symbol": record[1],
                    "timestamp": record[2],
                    "open_price": record[3],
                    "high_price": record[4],
                    "low_price": record[5],
                    "close_price": record[6],
                    "volume": record[7]
                }

                expected_checksum = self.calculate_data_checksum(record_data)
                actual_checksum = record[8]

                if expected_checksum != actual_checksum:
                    checksum_errors += 1

            conn.close()

            logger.info(f"トランザクション原子性テスト結果:")
            logger.info(f"  総トランザクション数: {num_transactions}")
            logger.info(f"  成功トランザクション: {transaction_stats['successful_transactions']}")
            logger.info(f"  失敗トランザクション: {transaction_stats['failed_transactions']}")
            logger.info(f"  ロールバック回数: {transaction_stats['rollback_count']}")
            logger.info(f"  市場データ件数: {market_data_count}")
            logger.info(f"  戦略結果件数: {strategy_results_count}")
            logger.info(f"  チェックサムエラー: {checksum_errors}")
            logger.info(f"  処理時間: {processing_time:.3f}秒")

            # ACID特性の確認
            assert market_data_count == expected_count, f"市場データ件数が不一致: {market_data_count} vs {expected_count}"
            assert strategy_results_count == expected_count, f"戦略結果件数が不一致: {strategy_results_count} vs {expected_count}"
            assert checksum_errors == 0, f"データ整合性エラーが発生: {checksum_errors}件"
            assert transaction_stats["rollback_count"] == transaction_stats["failed_transactions"], "ロールバック回数が不一致"

            logger.info("✅ トランザクション原子性テスト成功")

        except Exception as e:
            pytest.fail(f"トランザクション原子性テストエラー: {e}")

    def test_database_lock_contention(self):
        """テスト57: データベースロック競合時の処理"""
        logger.info("🔍 データベースロック競合テスト開始")

        try:
            # テスト用データベース
            test_db = os.path.join(self.temp_dir, "lock_test.db")
            self.create_test_database(test_db)

            # ロック競合統計
            lock_stats = {
                "read_operations": 0,
                "write_operations": 0,
                "lock_timeouts": 0,
                "deadlocks": 0,
                "successful_operations": 0,
                "failed_operations": 0
            }

            def heavy_read_operation(thread_id: int) -> Dict:
                """重い読み取り操作"""
                conn = sqlite3.connect(test_db, timeout=5.0)
                cursor = conn.cursor()

                try:
                    start_time = time.time()

                    # 複雑な読み取りクエリ
                    cursor.execute("""
                        SELECT symbol, AVG(close_price), COUNT(*), MIN(timestamp), MAX(timestamp)
                        FROM market_data
                        GROUP BY symbol
                    """)
                    results = cursor.fetchall()

                    # 追加の読み取り操作
                    cursor.execute("SELECT COUNT(*) FROM strategy_results")
                    count = cursor.fetchone()[0]

                    execution_time = time.time() - start_time
                    lock_stats["read_operations"] += 1
                    lock_stats["successful_operations"] += 1

                    return {
                        "thread_id": thread_id,
                        "operation": "read",
                        "success": True,
                        "execution_time": execution_time,
                        "results_count": len(results)
                    }

                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e).lower():
                        lock_stats["lock_timeouts"] += 1
                    lock_stats["failed_operations"] += 1

                    return {
                        "thread_id": thread_id,
                        "operation": "read",
                        "success": False,
                        "error": str(e)
                    }

                finally:
                    conn.close()

            def heavy_write_operation(thread_id: int) -> Dict:
                """重い書き込み操作"""
                conn = sqlite3.connect(test_db, timeout=5.0)
                cursor = conn.cursor()

                try:
                    start_time = time.time()

                    # 複数のレコードを挿入
                    for i in range(10):
                        # thread_idが文字列の場合は数値に変換
                        thread_id_num = int(thread_id.split('_')[-1]) if isinstance(thread_id, str) else thread_id
                        timestamp = int(time.time()) + thread_id_num * 1000 + i

                        market_data = {
                            "symbol": "BTC/USDT",
                            "timestamp": timestamp,
                            "open_price": 50000 + np.random.normal(0, 100),
                            "high_price": 50000 + np.random.normal(0, 100) + 50,
                            "low_price": 50000 + np.random.normal(0, 100) - 50,
                            "close_price": 50000 + np.random.normal(0, 100),
                            "volume": np.random.exponential(1000)
                        }

                        checksum = self.calculate_data_checksum(market_data)

                        cursor.execute("""
                            INSERT INTO market_data (symbol, timestamp, open_price, high_price, low_price, close_price, volume, checksum)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (market_data["symbol"], market_data["timestamp"], market_data["open_price"],
                             market_data["high_price"], market_data["low_price"], market_data["close_price"],
                             market_data["volume"], checksum))

                    conn.commit()
                    execution_time = time.time() - start_time
                    lock_stats["write_operations"] += 1
                    lock_stats["successful_operations"] += 1

                    return {
                        "thread_id": thread_id,
                        "operation": "write",
                        "success": True,
                        "execution_time": execution_time,
                        "records_inserted": 10
                    }

                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e).lower():
                        lock_stats["lock_timeouts"] += 1
                    elif "deadlock" in str(e).lower():
                        lock_stats["deadlocks"] += 1
                    lock_stats["failed_operations"] += 1

                    return {
                        "thread_id": thread_id,
                        "operation": "write",
                        "success": False,
                        "error": str(e)
                    }

                finally:
                    conn.close()

            # 初期データを挿入
            conn = sqlite3.connect(test_db)
            cursor = conn.cursor()

            for i in range(100):
                timestamp = int(time.time()) - 1000 + i
                market_data = {
                    "symbol": "BTC/USDT",
                    "timestamp": timestamp,
                    "open_price": 50000,
                    "high_price": 50100,
                    "low_price": 49900,
                    "close_price": 50000,
                    "volume": 1000
                }

                checksum = self.calculate_data_checksum(market_data)

                cursor.execute("""
                    INSERT INTO market_data (symbol, timestamp, open_price, high_price, low_price, close_price, volume, checksum)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (market_data["symbol"], market_data["timestamp"], market_data["open_price"],
                     market_data["high_price"], market_data["low_price"], market_data["close_price"],
                     market_data["volume"], checksum))

            conn.commit()
            conn.close()

            # 並行操作でロック競合をシミュレート
            num_read_threads = 5
            num_write_threads = 3

            start_time = time.time()

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = []

                # 読み取りスレッド
                for i in range(num_read_threads):
                    future = executor.submit(heavy_read_operation, f"read_{i}")
                    futures.append(future)

                # 書き込みスレッド
                for i in range(num_write_threads):
                    future = executor.submit(heavy_write_operation, f"write_{i}")
                    futures.append(future)

                results = [future.result() for future in futures]

            total_time = time.time() - start_time

            # 結果分析
            successful_reads = sum(1 for r in results if r["operation"] == "read" and r["success"])
            successful_writes = sum(1 for r in results if r["operation"] == "write" and r["success"])
            failed_operations = sum(1 for r in results if not r["success"])

            logger.info(f"データベースロック競合テスト結果:")
            logger.info(f"  総操作数: {len(results)}")
            logger.info(f"  成功した読み取り: {successful_reads}/{num_read_threads}")
            logger.info(f"  成功した書き込み: {successful_writes}/{num_write_threads}")
            logger.info(f"  失敗した操作: {failed_operations}")
            logger.info(f"  ロックタイムアウト: {lock_stats['lock_timeouts']}")
            logger.info(f"  デッドロック: {lock_stats['deadlocks']}")
            logger.info(f"  総処理時間: {total_time:.3f}秒")

            # ロック競合処理の確認
            success_rate = (successful_reads + successful_writes) / len(results)
            assert success_rate >= 0.7, f"成功率が低すぎます: {success_rate:.1%}"
            assert lock_stats["deadlocks"] == 0, f"デッドロックが発生しました: {lock_stats['deadlocks']}"

            # データ整合性の最終確認
            conn = sqlite3.connect(test_db)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM market_data")
            final_count = cursor.fetchone()[0]

            expected_min_count = 100 + (successful_writes * 10)  # 初期100 + 書き込み成功数 * 10
            assert final_count >= expected_min_count, f"データ件数が期待値を下回ります: {final_count} < {expected_min_count}"

            conn.close()

            logger.info("✅ データベースロック競合テスト成功")

        except Exception as e:
            pytest.fail(f"データベースロック競合テストエラー: {e}")


if __name__ == "__main__":
    # テスト実行
    test_instance = TestDataConsistency()
    
    tests = [
        test_instance.test_multi_source_data_consistency,
        test_instance.test_backup_restore_consistency,
        test_instance.test_distributed_processing_synchronization,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test_instance.setup_method()
            test()
            test_instance.teardown_method()
            passed += 1
        except Exception as e:
            logger.error(f"テスト失敗: {test.__name__}: {e}")
            failed += 1
    
    print(f"\n📊 データ整合性・一貫性テスト結果: 成功 {passed}, 失敗 {failed}")
    print(f"成功率: {passed / (passed + failed) * 100:.1f}%")
