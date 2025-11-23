import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

from app.services.ml.ml_training_service import MLTrainingService
from app.config.unified_config import unified_config

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.append(str(backend_dir))



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_data(n_rows=1000):
    dates = pd.date_range(start='2024-01-01', periods=n_rows, freq='4h')
    data = {
        'open': np.random.rand(n_rows) * 100 + 10000,
        'high': np.random.rand(n_rows) * 100 + 10100,
        'low': np.random.rand(n_rows) * 100 + 9900,
        'close': np.random.rand(n_rows) * 100 + 10000,
        'volume': np.random.rand(n_rows) * 1000,
    }
    df = pd.DataFrame(data, index=dates)
    # Ensure high/low logic
    df['high'] = df[['open', 'close']].max(axis=1) + df['high'] * 0.01
    df['low'] = df[['open', 'close']].min(axis=1) - df['low'] * 0.01
    return df

def verify_pipeline():
    logger.info("🚀 Starting Volatility Pipeline Verification")

    # 1. Create Dummy Data
    df = create_dummy_data()
    logger.info(f"Created dummy data: {df.shape}")

    # 2. Initialize Service
    service = MLTrainingService(trainer_type="single")
    
    # 3. Configure for Volatility Prediction
    unified_config.ml.training.label_generation.use_preset = True
    unified_config.ml.training.label_generation.default_preset = "volatility_4h_14bars"
    
    logger.info(f"Using preset: {unified_config.ml.training.label_generation.default_preset}")

    # 4. Train Model
    logger.info("Starting training...")
    try:
        result = service.train_model(
            training_data=df,
            save_model=False,
            model_name="test_volatility_model",
            # Force lightgbm for speed
            single_model_config={"model_type": "lightgbm"}
        )
        
        logger.info("Training completed.")
        logger.info(f"Result keys: {result.keys()}")
        
        # 5. Verify Model Output
        if "accuracy" in result:
            logger.info(f"Accuracy: {result['accuracy']}")
        
        # 6. Verify Prediction
        logger.info("Verifying prediction...")
        # Use the last few rows for prediction (simulate new data)
        # Note: evaluate_model expects enough data to calculate features
        test_features = df.iloc[-100:] 
        
        eval_result = service.evaluate_model(test_data=test_features)
        logger.info(f"Evaluation Result Keys: {eval_result.keys()}")
        
        predictions = eval_result.get("predictions")
        logger.info(f"Predictions: {predictions}")
        
        # Check keys
        if isinstance(predictions, dict):
            keys = predictions.keys()
            logger.info(f"Prediction Keys: {keys}")
            if "trend" in keys and "range" in keys:
                logger.info("✅ Verification SUCCESS: 'trend' and 'range' keys found.")
            else:
                logger.error("❌ Verification FAILED: Missing 'trend' or 'range' keys.")
                sys.exit(1)
        else:
            logger.error(f"❌ Verification FAILED: Predictions is not a dict: {type(predictions)}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Verification FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    verify_pipeline()
