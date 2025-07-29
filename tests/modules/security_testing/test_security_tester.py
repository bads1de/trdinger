"""
Unit tests for SecurityTester module.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

try:
    from ..security_testing.security_tester import (
        SecurityTester,
        SecurityViolationType,
        SecurityViolation,
    )
    from ...config.test_config import TestConfig, SecurityTestConfig
    from ...orchestrator.test_orchestrator import TestStatus
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from modules.security_testing.security_tester import (
        SecurityTester,
        SecurityViolationType,
        SecurityViolation,
    )
    from config.test_config import TestConfig, SecurityTestConfig
    from orchestrator.test_orchestrator import TestStatus


class TestSecurityTester:
    """Test cases for SecurityTester class."""

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

    def test_get_module_name(self):
        """Test that module name is returned correctly."""
        assert self.security_tester.get_module_name() == "security_testing"

    @pytest.mark.asyncio
    async def test_api_key_exposure_detection(self):
        """Test API key exposure detection functionality."""
        # Create temporary file with API key
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('api_key = "sk-1234567890abcdef1234567890abcdef"\n')
            f.write('secret_key = "secret_abcdef1234567890abcdef1234567890"\n')
            temp_file = Path(f.name)

        try:
            violations = []
            await self.security_tester._scan_directory_for_secrets(
                temp_file.parent, violations
            )

            # Should detect API key exposure
            assert len(violations) >= 2
            assert any(
                v.violation_type == SecurityViolationType.API_KEY_EXPOSURE
                for v in violations
            )

        finally:
            temp_file.unlink()

    @pytest.mark.asyncio
    async def test_input_validation(self):
        """Test input validation against malicious inputs."""
        result = await self.security_tester._test_input_validation()

        assert result.test_cases_run > 0
        assert isinstance(result.malicious_inputs_blocked, int)
        assert isinstance(result.validation_failures, int)
        assert result.test_duration_seconds > 0

    @pytest.mark.asyncio
    async def test_data_encryption_verification(self):
        """Test data encryption verification."""
        result = await self.security_tester._test_data_encryption()

        assert result.encryption_tests_run > 0
        assert isinstance(result.data_properly_encrypted, bool)
        assert isinstance(result.weak_encryption_detected, bool)
        assert result.test_duration_seconds > 0

    @pytest.mark.asyncio
    async def test_log_file_security_scanning(self):
        """Test log file security scanning."""
        # Create temporary log file with sensitive data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("INFO: User logged in\n")
            f.write("DEBUG: api_key=sk-1234567890abcdef1234567890abcdef\n")
            temp_log = Path(f.name)

        try:
            # Update config to scan the temp log
            self.security_config.log_scan_paths = [str(temp_log)]

            result = await self.security_tester._test_log_file_security()

            assert result.log_files_scanned >= 1
            assert result.compromised_log_files >= 1
            assert len(result.sensitive_data_exposures) >= 1

        finally:
            temp_log.unlink()

    @pytest.mark.asyncio
    async def test_run_tests_success(self):
        """Test successful execution of all security tests."""
        result = await self.security_tester.run_tests()

        assert result.module_name == "security_testing"
        assert result.status in [TestStatus.COMPLETED, TestStatus.FAILED]
        assert result.tests_run > 0
        assert result.execution_time_seconds > 0
        assert result.start_time is not None
        assert result.end_time is not None
        assert "api_key_exposure" in result.detailed_results
        assert "input_validation" in result.detailed_results
        assert "data_encryption" in result.detailed_results
        assert "log_security" in result.detailed_results

    @pytest.mark.asyncio
    async def test_run_tests_with_exception(self):
        """Test handling of exceptions during test execution."""
        # Mock a method to raise an exception
        with patch.object(
            self.security_tester,
            "_test_api_key_exposure",
            side_effect=Exception("Test error"),
        ):
            result = await self.security_tester.run_tests()

            assert result.status == TestStatus.FAILED
            assert len(result.error_messages) > 0
            assert "Test error" in str(result.error_messages)
            assert result.exception_details is not None

    def test_security_violation_creation(self):
        """Test SecurityViolation dataclass creation."""
        violation = SecurityViolation(
            violation_type=SecurityViolationType.API_KEY_EXPOSURE,
            severity="critical",
            description="Test violation",
            location="test_file.py",
            evidence="api_key = 'secret'",
            recommendation="Remove API key",
        )

        assert violation.violation_type == SecurityViolationType.API_KEY_EXPOSURE
        assert violation.severity == "critical"
        assert violation.description == "Test violation"
        assert violation.location == "test_file.py"
        assert violation.evidence == "api_key = 'secret'"
        assert violation.recommendation == "Remove API key"
        assert violation.timestamp is not None


if __name__ == "__main__":
    # Run tests directly
    async def run_tests():
        test_instance = TestSecurityTester()
        test_instance.setup_method()

        print("Testing SecurityTester module...")

        # Test module name
        module_name = test_instance.security_tester.get_module_name()
        print(f"Module name: {module_name}")

        # Test input validation
        await test_instance.test_input_validation()
        print("Input validation test passed")

        # Test data encryption
        await test_instance.test_data_encryption_verification()
        print("Data encryption test passed")

        # Test full run
        await test_instance.test_run_tests_success()
        print("Full test run passed")

        print("All tests completed successfully!")

    asyncio.run(run_tests())
