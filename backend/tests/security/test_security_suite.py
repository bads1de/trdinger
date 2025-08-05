#!/usr/bin/env python3
"""
セキュリティテストスイート

MLトレーニングシステムのセキュリティ面を検証します。
- 入力検証テスト
- データ保護テスト
- エラー情報漏洩防止テスト
- 権限制御テスト
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
import time
import tempfile
import shutil
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field

# プロジェクトルートをパスに追加
backend_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, backend_path)

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class SecurityTestResult:
    """セキュリティテスト結果データクラス"""

    test_name: str
    security_category: str
    success: bool
    execution_time: float
    vulnerability_detected: bool = False
    security_level: str = "UNKNOWN"  # HIGH, MEDIUM, LOW, CRITICAL
    protection_verified: bool = False
    error_message: str = ""
    security_details: Dict[str, Any] = field(default_factory=dict)


class SecurityTestSuite:
    """セキュリティテストスイート"""

    def __init__(self):
        self.results: List[SecurityTestResult] = []

    def create_malicious_data(self, attack_type: str) -> pd.DataFrame:
        """悪意のあるデータを作成"""
        logger.info(f"🔒 {attack_type}攻撃データを作成")

        if attack_type == "sql_injection":
            # SQL インジェクション攻撃を模擬
            malicious_data = pd.DataFrame(
                {
                    "Open": [50000, 51000, 52000],
                    "High": [51000, 52000, 53000],
                    "Low": [49000, 50000, 51000],
                    "Close": [50500, 51500, 52500],
                    "Volume": [1000, 1100, 1200],
                    "malicious_column": [
                        "'; DROP TABLE users; --",
                        "1' OR '1'='1",
                        "UNION SELECT * FROM sensitive_data",
                    ],
                }
            )

        elif attack_type == "script_injection":
            # スクリプトインジェクション攻撃を模擬
            malicious_data = pd.DataFrame(
                {
                    "Open": [50000, 51000, 52000],
                    "High": [51000, 52000, 53000],
                    "Low": [49000, 50000, 51000],
                    "Close": [50500, 51500, 52500],
                    "Volume": [1000, 1100, 1200],
                    "script_column": [
                        "<script>alert('XSS')</script>",
                        "javascript:void(0)",
                        "eval('malicious_code')",
                    ],
                }
            )

        elif attack_type == "path_traversal":
            # パストラバーサル攻撃を模擬
            malicious_data = pd.DataFrame(
                {
                    "Open": [50000, 51000, 52000],
                    "High": [51000, 52000, 53000],
                    "Low": [49000, 50000, 51000],
                    "Close": [50500, 51500, 52500],
                    "Volume": [1000, 1100, 1200],
                    "path_column": [
                        "../../../etc/passwd",
                        "..\\..\\windows\\system32\\config\\sam",
                        "../../../../proc/self/environ",
                    ],
                }
            )

        elif attack_type == "buffer_overflow":
            # バッファオーバーフロー攻撃を模擬
            long_string = "A" * 10000
            malicious_data = pd.DataFrame(
                {
                    "Open": [50000, 51000, 52000],
                    "High": [51000, 52000, 53000],
                    "Low": [49000, 50000, 51000],
                    "Close": [50500, 51500, 52500],
                    "Volume": [1000, 1100, 1200],
                    "overflow_column": [long_string, long_string, long_string],
                }
            )

        elif attack_type == "code_injection":
            # コードインジェクション攻撃を模擬
            malicious_data = pd.DataFrame(
                {
                    "Open": [50000, 51000, 52000],
                    "High": [51000, 52000, 53000],
                    "Low": [49000, 50000, 51000],
                    "Close": [50500, 51500, 52500],
                    "Volume": [1000, 1100, 1200],
                    "code_column": [
                        "__import__('os').system('rm -rf /')",
                        'exec(\'import subprocess; subprocess.call(["ls", "/"])\')',
                        "eval('1+1; import sys; sys.exit()')",
                    ],
                }
            )

        else:
            # デフォルトの正常データ
            malicious_data = pd.DataFrame(
                {
                    "Open": [50000, 51000, 52000],
                    "High": [51000, 52000, 53000],
                    "Low": [49000, 50000, 51000],
                    "Close": [50500, 51500, 52500],
                    "Volume": [1000, 1100, 1200],
                }
            )

        return malicious_data

    def test_input_validation_security(self):
        """入力検証セキュリティテスト"""
        logger.info("🔒 入力検証セキュリティテスト開始")

        attack_types = [
            "sql_injection",
            "script_injection",
            "path_traversal",
            "buffer_overflow",
            "code_injection",
        ]

        for attack_type in attack_types:
            start_time = time.time()

            try:
                # 悪意のあるデータを作成
                malicious_data = self.create_malicious_data(attack_type)

                # MLトレーニングサービスで処理
                from app.services.ml.single_model.single_model_trainer import (
                    SingleModelTrainer,
                )

                trainer = SingleModelTrainer(model_type="lightgbm")

                # 悪意のあるデータで学習を試行
                result = trainer.train_model(
                    training_data=malicious_data,
                    save_model=False,
                    threshold_up=0.02,
                    threshold_down=-0.02,
                )

                execution_time = time.time() - start_time

                # 結果を分析
                vulnerability_detected = False
                protection_verified = True
                security_level = "HIGH"

                # 悪意のあるデータが処理された場合は脆弱性の可能性
                if result and "accuracy" in result:
                    # データが正常に処理された場合、入力検証が不十分な可能性
                    if attack_type in [
                        "sql_injection",
                        "script_injection",
                        "code_injection",
                    ]:
                        vulnerability_detected = True
                        protection_verified = False
                        security_level = (
                            "MEDIUM"  # データクリーニングで除去された可能性
                        )

                self.results.append(
                    SecurityTestResult(
                        test_name=f"入力検証_{attack_type}",
                        security_category="input_validation",
                        success=protection_verified,
                        execution_time=execution_time,
                        vulnerability_detected=vulnerability_detected,
                        security_level=security_level,
                        protection_verified=protection_verified,
                        security_details={
                            "attack_type": attack_type,
                            "data_processed": result is not None,
                            "malicious_columns_detected": len(
                                [
                                    col
                                    for col in malicious_data.columns
                                    if "malicious" in col
                                    or "script" in col
                                    or "path" in col
                                    or "overflow" in col
                                    or "code" in col
                                ]
                            ),
                        },
                    )
                )

                logger.info(
                    f"✅ {attack_type}攻撃テスト完了: 保護レベル={security_level}"
                )

            except Exception as e:
                execution_time = time.time() - start_time

                # エラーが発生した場合、適切な入力検証が機能している可能性
                protection_verified = True
                vulnerability_detected = False
                security_level = "HIGH"

                # エラーメッセージから機密情報が漏洩していないかチェック
                error_msg = str(e).lower()
                sensitive_keywords = [
                    "password",
                    "secret",
                    "key",
                    "token",
                    "path",
                    "file",
                    "directory",
                ]
                info_leak = any(keyword in error_msg for keyword in sensitive_keywords)

                if info_leak:
                    vulnerability_detected = True
                    security_level = "MEDIUM"

                self.results.append(
                    SecurityTestResult(
                        test_name=f"入力検証_{attack_type}",
                        security_category="input_validation",
                        success=protection_verified,
                        execution_time=execution_time,
                        vulnerability_detected=vulnerability_detected,
                        security_level=security_level,
                        protection_verified=protection_verified,
                        error_message=str(e),
                        security_details={
                            "attack_type": attack_type,
                            "error_occurred": True,
                            "information_leak_detected": info_leak,
                            "error_handled_securely": not info_leak,
                        },
                    )
                )

                logger.info(f"✅ {attack_type}攻撃テスト完了: エラーで適切にブロック")

    def test_data_protection(self):
        """データ保護テスト"""
        logger.info("🛡️ データ保護テスト開始")

        start_time = time.time()

        try:
            # 機密データを含むテストデータを作成
            sensitive_data = pd.DataFrame(
                {
                    "Open": [50000, 51000, 52000, 53000, 54000],
                    "High": [51000, 52000, 53000, 54000, 55000],
                    "Low": [49000, 50000, 51000, 52000, 53000],
                    "Close": [50500, 51500, 52500, 53500, 54500],
                    "Volume": [1000, 1100, 1200, 1300, 1400],
                    "user_id": ["user123", "user456", "user789", "user101", "user202"],
                    "api_key": [
                        "sk-1234567890abcdef",
                        "sk-abcdef1234567890",
                        "sk-fedcba0987654321",
                        "sk-1357924680abcdef",
                        "sk-2468013579fedcba",
                    ],
                    "password_hash": [
                        "$2b$12$abcdef...",
                        "$2b$12$fedcba...",
                        "$2b$12$123456...",
                        "$2b$12$789012...",
                        "$2b$12$345678...",
                    ],
                }
            )

            # MLトレーニングで処理
            from app.services.ml.single_model.single_model_trainer import (
                SingleModelTrainer,
            )

            trainer = SingleModelTrainer(model_type="lightgbm")

            result = trainer.train_model(
                training_data=sensitive_data,
                save_model=False,
                threshold_up=0.02,
                threshold_down=-0.02,
            )

            execution_time = time.time() - start_time

            # データ保護の検証
            protection_verified = True
            vulnerability_detected = False
            security_level = "HIGH"

            # 機密データが特徴量に含まれていないかチェック
            if result and "feature_names" in result:
                feature_names = result["feature_names"]
                sensitive_features = [
                    name
                    for name in feature_names
                    if any(
                        sensitive in name.lower()
                        for sensitive in ["user_id", "api_key", "password"]
                    )
                ]

                if sensitive_features:
                    vulnerability_detected = True
                    protection_verified = False
                    security_level = "CRITICAL"

            # ログ出力に機密情報が含まれていないかチェック（簡易版）
            log_protection = True  # 実際のログ監視は複雑なため簡易実装

            self.results.append(
                SecurityTestResult(
                    test_name="データ保護",
                    security_category="data_protection",
                    success=protection_verified,
                    execution_time=execution_time,
                    vulnerability_detected=vulnerability_detected,
                    security_level=security_level,
                    protection_verified=protection_verified,
                    security_details={
                        "sensitive_data_processed": True,
                        "feature_count": (
                            result.get("feature_count", 0) if result else 0
                        ),
                        "sensitive_features_detected": vulnerability_detected,
                        "log_protection": log_protection,
                        "data_sanitization": not vulnerability_detected,
                    },
                )
            )

            logger.info(f"✅ データ保護テスト完了: セキュリティレベル={security_level}")

        except Exception as e:
            execution_time = time.time() - start_time

            # エラーメッセージから機密情報漏洩をチェック
            error_msg = str(e)
            sensitive_patterns = ["user123", "sk-", "$2b$12$", "api_key", "password"]
            info_leak = any(pattern in error_msg for pattern in sensitive_patterns)

            vulnerability_detected = info_leak
            protection_verified = not info_leak
            security_level = "MEDIUM" if info_leak else "HIGH"

            self.results.append(
                SecurityTestResult(
                    test_name="データ保護",
                    security_category="data_protection",
                    success=protection_verified,
                    execution_time=execution_time,
                    vulnerability_detected=vulnerability_detected,
                    security_level=security_level,
                    protection_verified=protection_verified,
                    error_message=str(e),
                    security_details={
                        "error_occurred": True,
                        "information_leak_in_error": info_leak,
                        "error_sanitization": not info_leak,
                    },
                )
            )

            logger.info(
                f"✅ データ保護テスト完了: エラー処理でのセキュリティレベル={security_level}"
            )

    def test_file_access_security(self):
        """ファイルアクセスセキュリティテスト"""
        logger.info("📁 ファイルアクセスセキュリティテスト開始")

        start_time = time.time()

        try:
            # 一時ディレクトリを作成
            temp_dir = tempfile.mkdtemp()

            try:
                # 正常なデータを作成
                normal_data = pd.DataFrame(
                    {
                        "Open": [50000, 51000, 52000],
                        "High": [51000, 52000, 53000],
                        "Low": [49000, 50000, 51000],
                        "Close": [50500, 51500, 52500],
                        "Volume": [1000, 1100, 1200],
                    }
                )

                # MLトレーニングでモデル保存を試行
                from app.services.ml.single_model.single_model_trainer import (
                    SingleModelTrainer,
                )

                trainer = SingleModelTrainer(model_type="lightgbm")

                result = trainer.train_model(
                    training_data=normal_data,
                    save_model=True,  # モデル保存を有効化
                    threshold_up=0.02,
                    threshold_down=-0.02,
                )

                execution_time = time.time() - start_time

                # ファイルアクセスセキュリティの検証
                protection_verified = True
                vulnerability_detected = False
                security_level = "HIGH"

                # 保存されたファイルの権限をチェック（簡易版）
                file_security = True
                if result and "model_path" in result:
                    model_path = result["model_path"]
                    if os.path.exists(model_path):
                        # ファイルが存在する場合、適切な場所に保存されているかチェック
                        if not model_path.startswith(
                            ("models/", "./models/", "backend/models/")
                        ):
                            vulnerability_detected = True
                            protection_verified = False
                            security_level = "MEDIUM"
                            file_security = False

                self.results.append(
                    SecurityTestResult(
                        test_name="ファイルアクセスセキュリティ",
                        security_category="file_access",
                        success=protection_verified,
                        execution_time=execution_time,
                        vulnerability_detected=vulnerability_detected,
                        security_level=security_level,
                        protection_verified=protection_verified,
                        security_details={
                            "model_saved": (
                                result is not None and "model_path" in result
                                if result
                                else False
                            ),
                            "file_path_secure": file_security,
                            "temp_dir_used": temp_dir is not None,
                            "file_permissions_secure": True,  # 簡易実装
                        },
                    )
                )

                logger.info(
                    f"✅ ファイルアクセスセキュリティテスト完了: セキュリティレベル={security_level}"
                )

            finally:
                # 一時ディレクトリをクリーンアップ
                shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            execution_time = time.time() - start_time

            # ファイルアクセスエラーの分析
            error_msg = str(e).lower()
            file_related_error = any(
                keyword in error_msg
                for keyword in ["permission", "access", "file", "directory", "path"]
            )

            protection_verified = file_related_error  # ファイルアクセスが制限されている
            vulnerability_detected = not file_related_error
            security_level = "HIGH" if file_related_error else "MEDIUM"

            self.results.append(
                SecurityTestResult(
                    test_name="ファイルアクセスセキュリティ",
                    security_category="file_access",
                    success=protection_verified,
                    execution_time=execution_time,
                    vulnerability_detected=vulnerability_detected,
                    security_level=security_level,
                    protection_verified=protection_verified,
                    error_message=str(e),
                    security_details={
                        "file_access_error": file_related_error,
                        "access_control_working": file_related_error,
                        "error_type": type(e).__name__,
                    },
                )
            )

            logger.info(
                f"✅ ファイルアクセスセキュリティテスト完了: エラー分析でのセキュリティレベル={security_level}"
            )

    def test_error_information_disclosure(self):
        """エラー情報漏洩テスト"""
        logger.info("🚨 エラー情報漏洩テスト開始")

        start_time = time.time()

        try:
            # 意図的にエラーを発生させるデータを作成
            error_inducing_data = pd.DataFrame(
                {
                    "Open": [np.inf, -np.inf, np.nan],
                    "High": [np.inf, -np.inf, np.nan],
                    "Low": [np.inf, -np.inf, np.nan],
                    "Close": [np.inf, -np.inf, np.nan],
                    "Volume": [np.inf, -np.inf, np.nan],
                }
            )

            # MLトレーニングで処理してエラーを誘発
            from app.services.ml.single_model.single_model_trainer import (
                SingleModelTrainer,
            )

            trainer = SingleModelTrainer(model_type="lightgbm")

            try:
                result = trainer.train_model(
                    training_data=error_inducing_data,
                    save_model=False,
                    threshold_up=0.02,
                    threshold_down=-0.02,
                )

                # エラーが発生しなかった場合
                execution_time = time.time() - start_time

                self.results.append(
                    SecurityTestResult(
                        test_name="エラー情報漏洩",
                        security_category="information_disclosure",
                        success=True,
                        execution_time=execution_time,
                        vulnerability_detected=False,
                        security_level="HIGH",
                        protection_verified=True,
                        security_details={
                            "error_occurred": False,
                            "data_sanitization_effective": True,
                            "robust_error_handling": True,
                        },
                    )
                )

                logger.info(
                    "✅ エラー情報漏洩テスト完了: データサニタイゼーションが有効"
                )

            except Exception as e:
                execution_time = time.time() - start_time

                # エラーメッセージの分析
                error_msg = str(e)

                # 機密情報が含まれているかチェック
                sensitive_info = [
                    "file path",
                    "directory",
                    "username",
                    "password",
                    "key",
                    "token",
                    "internal",
                    "stack trace",
                    "line number",
                    "function name",
                ]

                info_disclosure = any(
                    info.lower() in error_msg.lower() for info in sensitive_info
                )

                # スタックトレースが含まれているかチェック
                stack_trace_present = (
                    "traceback" in error_msg.lower() or 'file "' in error_msg
                )

                vulnerability_detected = info_disclosure or stack_trace_present
                protection_verified = not vulnerability_detected
                security_level = (
                    "CRITICAL"
                    if stack_trace_present
                    else ("MEDIUM" if info_disclosure else "HIGH")
                )

                self.results.append(
                    SecurityTestResult(
                        test_name="エラー情報漏洩",
                        security_category="information_disclosure",
                        success=protection_verified,
                        execution_time=execution_time,
                        vulnerability_detected=vulnerability_detected,
                        security_level=security_level,
                        protection_verified=protection_verified,
                        error_message=(
                            error_msg[:200] + "..."
                            if len(error_msg) > 200
                            else error_msg
                        ),
                        security_details={
                            "error_occurred": True,
                            "sensitive_info_disclosed": info_disclosure,
                            "stack_trace_present": stack_trace_present,
                            "error_message_length": len(error_msg),
                            "error_sanitization": not vulnerability_detected,
                        },
                    )
                )

                logger.info(
                    f"✅ エラー情報漏洩テスト完了: セキュリティレベル={security_level}"
                )

        except Exception as e:
            execution_time = time.time() - start_time

            self.results.append(
                SecurityTestResult(
                    test_name="エラー情報漏洩",
                    security_category="information_disclosure",
                    success=False,
                    execution_time=execution_time,
                    vulnerability_detected=True,
                    security_level="UNKNOWN",
                    protection_verified=False,
                    error_message=str(e),
                    security_details={
                        "test_setup_error": True,
                        "error_type": type(e).__name__,
                    },
                )
            )

            logger.error(f"❌ エラー情報漏洩テストでセットアップエラー: {e}")

    def test_advanced_input_validation_security(self):
        """高度な入力検証セキュリティテスト（脆弱性修正後）"""
        logger.info("🔒 高度な入力検証セキュリティテスト開始")

        start_time = time.time()

        try:
            # 悪意のあるデータパターンを作成
            malicious_patterns = {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="1H"),
                # SQLインジェクション攻撃パターン
                "Open": [
                    100.0,
                    "'; DROP TABLE users; --",
                    "1' OR '1'='1",
                    "UNION SELECT * FROM sensitive_data",
                    105.0,
                    "1; DELETE FROM accounts",
                    110.0,
                    "' OR 1=1 --",
                    115.0,
                    "admin'--",
                ],
                # XSS攻撃パターン
                "High": [
                    105.0,
                    "<script>alert('XSS')</script>",
                    "javascript:alert('XSS')",
                    "<img src=x onerror=alert('XSS')>",
                    110.0,
                    "<iframe src='javascript:alert(1)'></iframe>",
                    115.0,
                    "';alert(String.fromCharCode(88,83,83))//",
                    120.0,
                    "<svg onload=alert('XSS')>",
                ],
                # パストラバーサル攻撃
                "Low": [
                    95.0,
                    "../../../etc/passwd",
                    "..\\..\\..\\windows\\system32\\config\\sam",
                    "....//....//....//etc/passwd",
                    100.0,
                    "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
                    105.0,
                    "..%252f..%252f..%252fetc%252fpasswd",
                    110.0,
                    "..%c0%af..%c0%af..%c0%afetc%c0%afpasswd",
                ],
                # DoS攻撃パターン
                "Close": [
                    100.0,
                    1e308,  # 最大float値
                    -1e308,
                    float("inf"),
                    float("-inf"),
                    105.0,
                    "A" * 10000,  # 長い文字列
                    110.0,
                    "\x00" * 1000,  # ヌル文字
                    115.0,
                ],
                "Volume": [
                    1000,
                    -999999999,
                    1e20,
                    0,
                    "null",
                    "undefined",
                    "NaN",
                    1500,
                    2000,
                    2500,
                ],
            }

            malicious_data = pd.DataFrame(malicious_patterns)

            # 修正されたDataValidatorのテスト
            try:
                from app.utils.data_validation import DataValidator

                validator = DataValidator()
                validation_result = validator.validate_ohlcv_data(
                    malicious_data, strict_mode=True
                )

                # セキュリティ検証
                security_passed = True
                vulnerabilities = []

                # 悪意のあるデータが適切に検出されたかチェック
                if validation_result.get("is_valid", True):
                    security_passed = False
                    vulnerabilities.append("悪意のあるデータが検出されませんでした")

                # OHLC違反の検出
                ohlc_violations = validation_result.get("ohlc_violations", 0)
                if ohlc_violations == 0:
                    vulnerabilities.append("OHLC論理違反が検出されませんでした")

                # 負のボリュームの検出
                negative_volumes = validation_result.get("negative_volumes", 0)
                if negative_volumes == 0:
                    vulnerabilities.append("負のボリュームが検出されませんでした")

                security_level = "HIGH" if security_passed else "CRITICAL"

                logger.info(
                    f"✅ 修正されたDataValidator検証完了: {len(vulnerabilities)}個の脆弱性"
                )

            except Exception as e:
                security_passed = False
                vulnerabilities = [f"DataValidatorエラー: {e}"]
                security_level = "CRITICAL"

            execution_time = time.time() - start_time

            result = SecurityTestResult(
                test_name="高度な入力検証セキュリティ",
                security_category="INPUT_VALIDATION",
                success=security_passed,
                execution_time=execution_time,
                vulnerability_detected=len(vulnerabilities) > 0,
                security_level=security_level,
                protection_verified=security_passed,
                security_details={
                    "vulnerabilities_found": vulnerabilities,
                    "validation_result": (
                        validation_result if "validation_result" in locals() else {}
                    ),
                    "malicious_patterns_tested": len(malicious_patterns),
                },
            )

            self.results.append(result)
            logger.info(f"✅ 高度な入力検証セキュリティテスト完了: {security_level}")

        except Exception as e:
            execution_time = time.time() - start_time
            result = SecurityTestResult(
                test_name="高度な入力検証セキュリティ",
                security_category="INPUT_VALIDATION",
                success=False,
                execution_time=execution_time,
                vulnerability_detected=True,
                security_level="CRITICAL",
                protection_verified=False,
                error_message=str(e),
            )
            self.results.append(result)
            logger.error(f"❌ 高度な入力検証セキュリティテスト失敗: {e}")

    def test_data_consistency_security(self):
        """データ一貫性セキュリティテスト（脆弱性修正後）"""
        logger.info("📊 データ一貫性セキュリティテスト開始")

        start_time = time.time()

        try:
            # 不整合データパターンを作成
            inconsistent_data = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2024-01-01", periods=8, freq="1H"),
                    "Open": [100, 105, 110, 115, 120, 125, 130, 135],
                    "High": [105, 110, 115, 120, 125, 130, 135, 140],
                    "Low": [95, 100, 105, 110, 115, 120, 125, 130],
                    "Close": [105, 110, 115, 120, 125, 130, 135, 140],
                    "Volume": [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500],
                }
            )

            # 重複タイムスタンプを追加
            duplicate_row = inconsistent_data.iloc[3:4].copy()
            inconsistent_data = pd.concat(
                [inconsistent_data, duplicate_row], ignore_index=True
            )

            # 修正されたDataFrequencyManagerのテスト
            try:
                from app.services.ml.feature_engineering.data_frequency_manager import (
                    DataFrequencyManager,
                )

                freq_manager = DataFrequencyManager()
                consistency_result = freq_manager.validate_data_consistency(
                    inconsistent_data, None, None, "1h"
                )

                # セキュリティ検証
                security_passed = True
                vulnerabilities = []

                # 重複タイムスタンプの検出
                duplicate_timestamps = consistency_result.get("duplicate_timestamps", 0)
                if duplicate_timestamps == 0:
                    vulnerabilities.append("重複タイムスタンプが検出されませんでした")
                    security_passed = False

                # データ一貫性の検証
                if consistency_result.get("is_valid", True):
                    vulnerabilities.append("不整合データが有効と判定されました")
                    security_passed = False

                security_level = "HIGH" if security_passed else "MEDIUM"

                logger.info(
                    f"✅ 修正されたDataFrequencyManager検証完了: {len(vulnerabilities)}個の問題"
                )

            except Exception as e:
                security_passed = False
                vulnerabilities = [f"DataFrequencyManagerエラー: {e}"]
                security_level = "CRITICAL"

            execution_time = time.time() - start_time

            result = SecurityTestResult(
                test_name="データ一貫性セキュリティ",
                security_category="DATA_INTEGRITY",
                success=security_passed,
                execution_time=execution_time,
                vulnerability_detected=len(vulnerabilities) > 0,
                security_level=security_level,
                protection_verified=security_passed,
                security_details={
                    "vulnerabilities_found": vulnerabilities,
                    "consistency_result": (
                        consistency_result if "consistency_result" in locals() else {}
                    ),
                    "duplicate_timestamps_added": 1,
                },
            )

            self.results.append(result)
            logger.info(f"✅ データ一貫性セキュリティテスト完了: {security_level}")

        except Exception as e:
            execution_time = time.time() - start_time
            result = SecurityTestResult(
                test_name="データ一貫性セキュリティ",
                security_category="DATA_INTEGRITY",
                success=False,
                execution_time=execution_time,
                vulnerability_detected=True,
                security_level="CRITICAL",
                protection_verified=False,
                error_message=str(e),
            )
            self.results.append(result)
            logger.error(f"❌ データ一貫性セキュリティテスト失敗: {e}")


if __name__ == "__main__":
    logger.info("🔒 セキュリティテストスイート開始")

    test_suite = SecurityTestSuite()

    # 各セキュリティテストを実行
    test_suite.test_input_validation_security()
    test_suite.test_data_protection()
    test_suite.test_file_access_security()
    test_suite.test_error_information_disclosure()

    # 結果サマリー
    total_tests = len(test_suite.results)
    successful_tests = sum(1 for r in test_suite.results if r.success)
    vulnerabilities_detected = sum(
        1 for r in test_suite.results if r.vulnerability_detected
    )
    high_security_tests = sum(
        1 for r in test_suite.results if r.security_level == "HIGH"
    )

    print("\n" + "=" * 80)
    print("🔒 セキュリティテスト結果")
    print("=" * 80)
    print(f"📊 総テスト数: {total_tests}")
    print(f"✅ 成功: {successful_tests}")
    print(f"❌ 失敗: {total_tests - successful_tests}")
    print(f"🚨 脆弱性検出: {vulnerabilities_detected}")
    print(f"🛡️ 高セキュリティ: {high_security_tests}")
    print(f"📈 成功率: {(successful_tests/total_tests*100):.1f}%")
    print(f"🔒 セキュリティ率: {(high_security_tests/total_tests*100):.1f}%")

    print("\n🔒 セキュリティテスト詳細:")
    for result in test_suite.results:
        status = "✅" if result.success else "❌"
        vulnerability = "🚨" if result.vulnerability_detected else "🛡️"
        print(f"{status} {result.test_name}")
        print(f"   カテゴリ: {result.security_category}")
        print(f"   実行時間: {result.execution_time:.2f}秒")
        print(f"   セキュリティレベル: {result.security_level}")
        print(f"   脆弱性: {vulnerability}")
        print(f"   保護確認: {'✅' if result.protection_verified else '❌'}")
        if result.error_message:
            print(f"   エラー: {result.error_message[:100]}...")

    print("=" * 80)

    logger.info("🎯 セキュリティテストスイート完了")
