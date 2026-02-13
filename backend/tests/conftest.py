import pytest
import sys
import os

# Ensure backend path is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.utils.pandas_patch import patch_pandas_append


@pytest.fixture(scope="session", autouse=True)
def setup_pandas_patch():
    """Apply pandas patches for all tests."""
    patch_pandas_append()
