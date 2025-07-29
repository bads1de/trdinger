"""
Integration tests for SecurityTester with TestOrchestrator.
Tests the comprehensive security scanning system integration.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

try:
    from ..security_testing.security_tester import SecurityTester
    from ..security_testing.security_scanner import SecurityScanner
    from ...config.test_config import TestConfig, SecurityTestConfig
    from ...orchestrator.test_orchestrator import TestOrchestrator, TestStatus
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from modules.security_testing.security_tester import SecurityTester
    from modules.security_testing.security_scanner import SecurityScanner
    from config.test_config import TestConfig, SecurityTestConfig
    from orchestrator.test_orchestrator import TestOrchestrator, TestStatus


class TestSecurityIntegration:
    """Test cases for SecurityTester integration with TestOrchestrator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.security_config = SecurityTestConfig(
            sensitive_patterns=[
                r'api[_-]?key["\s]*[:=]["\s]*[a-zA-Z0-9]{20,}',
                r'secret[_-]?key["\s]*[:=]["\s]*[a-zA-Z0-9]{20,}',
            ],
            log_scan_paths=[],
            input_validation_test_cases=[],
            encryption_test_scenarios=[],
        )

        self.test_config = Mock(spec=TestConfig)
        self.test_config.security_test_config = self.security_config
        self.test_config.log_level = "INFO"

        self.security_tester = SecurityTester(self.test_config)

    @pytest.mark.asyncio
    async def test_orchestrator_integration(self):
        """Test SecurityTester integration with TestOrchestrator."""
        # Create test orchestrator
        orchestrator = TestOrchestrator(self.test_config)

        # Register SecurityTester
        orchestrator.register_test_module("security_testing", self.security_tester)

        # Verify registration
        registered_modules = orchestrator.get_registered_test_modules()
        assert "security_testing" in registered_modules

        # Run security tests through orchestrator
        results = await orchestrator.run_specific_tests(["security_testing"])

        assert results.overall_status in [TestStatus.COMPLETED, TestStatus.FAILED]
        assert "security_testing" in results.modules_results

        security_result = results.modules_results["security_testing"]
        assert security_result.module_name == "security_testing"
        assert security_result.tests_run > 0

    @pytest.mark.asyncio
    async def test_comprehensive_security_scan(self):
        """Test comprehensive security scanning functionality."""
        result = await self.security_tester._run_comprehensive_security_scan()

        assert result.automated_scan_completed is not None
        assert result.integration_successful is not None
        assert result.vulnerability_report is not None
        assert result.security_alerts_generated is not None
        assert result.scan_duration_seconds > 0

    @pytest.mark.asyncio
    async def test_security_scanner_standalone(self):
        """Test SecurityScanner as standalone component."""
        scanner = SecurityScanner(self.test_config)

        # Run comprehensive scan
        vulnerability_report = await scanner.run_comprehensive_scan()

        assert vulnerability_report.scan_id is not None
        assert vulnerability_report.total_files_scanned >= 0
        assert vulnerability_report.vulnerabilities_found >= 0
        assert vulnerability_report.scan_duration_seconds > 0
        assert isinstance(vulnerability_report.alerts_generated, list)

    @pytest.mark.asyncio
    async def test_security_alert_generation(self):
        """Test security alert generation system."""
        scanner = SecurityScanner(self.test_config)

        # Create mock alerts
        from modules.security_testing.security_scanner import (
            SecurityAlert,
            VulnerabilityType,
        )

        mock_alerts = [
            SecurityAlert(
                alert_id="test_alert_1",
                vulnerability_type=VulnerabilityType.HARDCODED_CREDENTIALS,
                severity="critical",
                title="Test Critical Alert",
                description="Test description",
                affected_files=["test.py"],
                evidence=["test evidence"],
                recommendations=["test recommendation"],
            ),
            SecurityAlert(
                alert_id="test_alert_2",
                vulnerability_type=VulnerabilityType.WEAK_CRYPTOGRAPHY,
                severity="high",
                title="Test High Alert",
                description="Test description",
                affected_files=["test.py"],
                evidence=["test evidence"],
                recommendations=["test recommendation"],
            ),
        ]

        # Generate alerts
        alert_summary = await scanner.generate_security_alerts(mock_alerts)

        assert alert_summary["total_alerts"] == 2
        assert alert_summary["critical_alerts"] == 1
        assert alert_summary["high_alerts"] == 1
        assert len(alert_summary["immediate_action_required"]) == 2

    def test_security_tester_module_interface(self):
        """Test that SecurityTester properly implements TestModuleInterface."""
        # Test module name
        assert self.security_tester.get_module_name() == "security_testing"

        # Test that run_tests method exists and is callable
        assert hasattr(self.security_tester, "run_tests")
        assert callable(getattr(self.security_tester, "run_tests"))

    @pytest.mark.asyncio
    async def test_vulnerability_detection_with_sample_files(self):
        """Test vulnerability detection with sample vulnerable files."""
        # Create temporary vulnerable files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create vulnerable Python file
            vulnerable_py = temp_path / "vulnerable.py"
            vulnerable_py.write_text(
                """
import hashlib

# Hardcoded API key (vulnerability)
api_key = "sk-1234567890abcdef1234567890abcdef"

# Weak cryptography (vulnerability)
password_hash = hashlib.md5(password.encode()).hexdigest()

# Code injection vulnerability
eval(user_input)
"""
            )

            # Create vulnerable JavaScript file
            vulnerable_js = temp_path / "vulnerable.js"
            vulnerable_js.write_text(
                """
// Hardcoded secret (vulnerability)
const secret_key = "secret_abcdef1234567890abcdef1234567890";

// Information disclosure (vulnerability)
console.log("API key:", api_key);
"""
            )

            # Run scanner on temporary directory
            scanner = SecurityScanner(self.test_config)

            # Mock the source paths to include our temp directory
            with patch.object(scanner, "_get_source_files") as mock_get_files:

                async def mock_file_generator(directory):
                    for file_path in temp_path.rglob("*"):
                        if file_path.is_file():
                            yield file_path

                mock_get_files.return_value = mock_file_generator(temp_path)

                alerts, files_scanned = (
                    await scanner._scan_source_code_vulnerabilities()
                )

                # Should detect vulnerabilities
                assert files_scanned >= 2
                assert len(alerts) >= 4  # At least 4 vulnerabilities should be detected


if __name__ == "__main__":
    # Run integration tests directly
    async def run_integration_tests():
        test_instance = TestSecurityIntegration()
        test_instance.setup_method()

        print("Testing SecurityTester integration...")

        # Test module interface
        test_instance.test_security_tester_module_interface()
        print("Module interface test passed")

        # Test comprehensive security scan
        await test_instance.test_comprehensive_security_scan()
        print("Comprehensive security scan test passed")

        # Test security scanner standalone
        await test_instance.test_security_scanner_standalone()
        print("Security scanner standalone test passed")

        # Test security alert generation
        await test_instance.test_security_alert_generation()
        print("Security alert generation test passed")

        # Test vulnerability detection
        await test_instance.test_vulnerability_detection_with_sample_files()
        print("Vulnerability detection test passed")

        print("All integration tests completed successfully!")

    asyncio.run(run_integration_tests())
