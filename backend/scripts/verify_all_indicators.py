import sys
import os
import pandas as pd
import numpy as np
import logging

# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
from app.services.indicators.config.indicator_config import (
    indicator_registry,
    initialize_all_indicators,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_dummy_data(length=300):
    """Generates dummy OHLCV data."""
    dates = pd.date_range(start="2023-01-01", periods=length, freq="D")

    # Generate some wave-like price action
    x = np.linspace(0, 4 * np.pi, length)
    base_price = 100 + 10 * np.sin(x) + np.random.normal(0, 1, length)

    open_prices = base_price
    close_prices = base_price + np.random.normal(0, 1.5, length)
    high_prices = np.maximum(open_prices, close_prices) + np.abs(
        np.random.normal(0, 1, length)
    )
    low_prices = np.minimum(open_prices, close_prices) - np.abs(
        np.random.normal(0, 1, length)
    )
    volumes = np.abs(np.random.normal(1000, 200, length))

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
        }
    )

    # Ensure all required columns are present (some indicators might look for uppercase)
    df["OPEN"] = df["open"]
    df["HIGH"] = df["high"]
    df["LOW"] = df["low"]
    df["CLOSE"] = df["close"]
    df["VOLUME"] = df["volume"]

    # Add extra columns for advanced features
    df["open_interest"] = np.abs(np.random.normal(100000, 10000, length))
    df["funding_rate"] = np.random.normal(0.0001, 0.00005, length)
    df["market_cap"] = df["close"] * 1000000

    return df


def verify_indicators():
    """Verifies all registered indicators."""

    # Initialize indicators explicitly to be sure
    initialize_all_indicators()

    service = TechnicalIndicatorService()
    df = generate_dummy_data(length=300)  # Ensure enough data for long periods

    all_indicators = indicator_registry.get_all_indicators()

    results = {"success": [], "failed": [], "errors": {}}

    print(f"Starting verification for {len(all_indicators)} indicators...")

    for name, config in all_indicators.items():
        # Skip aliases to avoid duplicate checking if desired, but checking aliases is also good.
        # Let's check unique indicator names to be concise
        if name != config.indicator_name:
            continue

        try:
            # Prepare default parameters
            params = {
                k: v.default_value
                for k, v in config.parameters.items()
                if hasattr(v, "default_value")
            }

            # If no default values in parameters, try config.default_values
            if not params and config.default_values:
                params = config.default_values.copy()

            # Filter params: remove keys that are in config.required_data
            # Also remove common data column names to avoid passing them as parameters (e.g. open=14)
            data_cols = {
                "open",
                "high",
                "low",
                "close",
                "volume",
                "open_interest",
                "funding_rate",
                "market_cap",
                "data",
            }
            # config.required_data might be empty or incomplete for pandas-ta wrappers, so use safe defaults
            forbidden_params = set(config.required_data) | data_cols

            clean_params = {}
            for k, v in params.items():
                if k not in forbidden_params:
                    clean_params[k] = v
            params = clean_params

            # Fallback for some common params if missing and likely needed
            if "length" not in params and "period" not in params:
                if any("length" in p for p in config.parameters):
                    params["length"] = 14
                elif any("period" in p for p in config.parameters):
                    params["period"] = 14

            logger.info(f"Testing {name} with params: {params}")

            result = service.calculate_indicator(df, name, params)

            # Check for validity
            if result is None:
                raise ValueError("Result is None")

            if isinstance(result, tuple):
                for i, r in enumerate(result):
                    if isinstance(r, (np.ndarray, pd.Series)):
                        if len(r) != len(df):
                            raise ValueError(
                                f"Result length mismatch at index {i}: got {len(r)}, expected {len(df)}"
                            )
                        if np.all(np.isnan(r)):
                            if name not in ["KST", "TRIX"]:
                                logger.warning(
                                    f"Indicator {name} produced all NaNs at index {i}"
                                )
                    else:
                        pass
            elif isinstance(result, (np.ndarray, pd.Series)):
                if len(result) != len(df):
                    raise ValueError(
                        f"Result length mismatch: got {len(result)}, expected {len(df)}"
                    )
                if np.all(np.isnan(result)):
                    logger.warning(f"Indicator {name} produced all NaNs")
            else:
                pass

            results["success"].append(name)

        except Exception as e:
            logger.error(f"Failed {name}: {str(e)}")
            results["failed"].append(name)
            results["errors"][name] = str(e)

    print("\n" + "=" * 50)
    print("VERIFICATION REPORT")
    print("=" * 50)
    print(f"Total Indicators: {len(results['success']) + len(results['failed'])}")
    print(f"Success: {len(results['success'])}")
    print(f"Failed: {len(results['failed'])}")

    if results["failed"]:
        print("\nFailed Indicators:")
        for name in results["failed"]:
            print(f"- {name}: {results['errors'][name]}")

    print("=" * 50)


if __name__ == "__main__":
    verify_indicators()
