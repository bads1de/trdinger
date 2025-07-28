"""
Comprehensive Testing Framework for Trdinger Trading Platform

This package provides a complete testing overhaul with the following modules:
- ML model accuracy testing and validation
- Backtest functionality comprehensive verification  
- Financial calculation precision testing with Decimal enforcement
- Concurrency and race condition testing
- Performance benchmarking and bottleneck detection
- Security testing and vulnerability scanning
- Comprehensive test reporting and CI/CD integration

Usage:
    from tests.orchestrator.test_orchestrator import TestOrchestrator
    from tests.config.test_config import get_test_config
    
    # Run all tests
    orchestrator = TestOrchestrator()
    results = await orchestrator.run_all_tests()
    
    # Run specific test modules
    results = await orchestrator.run_specific_tests(['ml_testing', 'financial_testing'])
"""

__version__ = "1.0.0"
__author__ = "Trdinger Development Team"

# Import main components for easy access
from .config.test_config import TestConfig, get_test_config
from .orchestrator.test_orchestrator import TestOrchestrator, TestResults, TestStatus
from .utils.cleanup_manager import TestCleanupManager, CleanupReport
from .utils.test_utilities import (
    TestLogger,
    DecimalHelper,
    MockDataGenerator,
    DatabaseTestHelper,
    FileSystemTestHelper,
    AsyncTestHelper,
    PerformanceTestHelper,
    SecurityTestHelper,
)
from .fixtures.test_fixtures import (
    get_ml_test_data,
    get_backtest_test_data,
    get_financial_test_scenarios,
    get_portfolio_test_data,
    get_performance_test_data,
    get_security_test_scenarios,
    cleanup_test_fixtures,
)

__all__ = [
    # Configuration
    "TestConfig",
    "get_test_config",
    # Orchestration
    "TestOrchestrator",
    "TestResults",
    "TestStatus",
    # Cleanup
    "TestCleanupManager",
    "CleanupReport",
    # Utilities
    "TestLogger",
    "DecimalHelper",
    "MockDataGenerator",
    "DatabaseTestHelper",
    "FileSystemTestHelper",
    "AsyncTestHelper",
    "PerformanceTestHelper",
    "SecurityTestHelper",
    # Fixtures
    "get_ml_test_data",
    "get_backtest_test_data",
    "get_financial_test_scenarios",
    "get_portfolio_test_data",
    "get_performance_test_data",
    "get_security_test_scenarios",
    "cleanup_test_fixtures",
]
