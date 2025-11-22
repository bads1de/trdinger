import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from unittest.mock import MagicMock

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.append(str(backend_dir))

from app.services.auto_strategy.core.hybrid_predictor import HybridPredictor
from app.services.ml.ml_training_service import MLTrainingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_data(n_rows=100):
    dates = pd.date_range(start='2024-01-01', periods=n_rows, freq='4h')
    data = {
        'open': np.random.rand(n_rows) * 100 + 10000,
        'high': np.random.rand(n_rows) * 100 + 10100,
        'low': np.random.rand(n_rows) * 100 + 9900,
        'close': np.random.rand(n_rows) * 100 + 10000,
        'volume': np.random.rand(n_rows) * 1000,
    }
    return pd.DataFrame(data, index=dates)

def verify_hybrid_predictor():
    logger.info("🚀 Starting HybridPredictor Verification")

    # 1. Create Dummy Data
    df = create_dummy_data()
    
    # 2. Mock MLTrainingService
    mock_service = MagicMock(spec=MLTrainingService)
    # Mock generate_signals to return volatility prediction
    mock_service.generate_signals.return_value = {"trend": 0.8, "range": 0.2}
    mock_service.get_training_status.return_value = {"is_trained": True}
    
    # Mock trainer attribute for is_trained check
    mock_trainer = MagicMock()
    mock_trainer.is_trained = True
    mock_service.trainer = mock_trainer

    # 3. Initialize HybridPredictor with mocked service class
    # We need to patch the _resolve_training_service_cls or pass the class
    # Since HybridPredictor instantiates the class, we pass a Mock class
    MockServiceClass = MagicMock(return_value=mock_service)
    MockServiceClass.get_available_single_models.return_value = ["lightgbm"]

    predictor = HybridPredictor(
        trainer_type="single",
        training_service_cls=MockServiceClass
    )
    
    # 4. Test Predict
    logger.info("Testing predict()...")
    try:
        prediction = predictor.predict(df)
        logger.info(f"Prediction result: {prediction}")
        
        if "trend" in prediction and "range" in prediction:
            logger.info("✅ Prediction keys correct.")
            if abs(prediction["trend"] + prediction["range"] - 1.0) < 1e-6:
                logger.info("✅ Probabilities sum to 1.0.")
            else:
                logger.error(f"❌ Probabilities do not sum to 1.0: {sum(prediction.values())}")
        else:
            logger.error(f"❌ Missing keys in prediction: {prediction.keys()}")
            
    except Exception as e:
        logger.error(f"❌ Predict failed: {e}")
        import traceback
        traceback.print_exc()

    # 5. Test Multi-Model Averaging (Volatility)
    logger.info("Testing multi-model averaging...")
    mock_service2 = MagicMock(spec=MLTrainingService)
    mock_service2.generate_signals.return_value = {"trend": 0.6, "range": 0.4}
    mock_service2.trainer = mock_trainer
    
    MockServiceClassMulti = MagicMock(side_effect=[mock_service, mock_service2])
    
    predictor_multi = HybridPredictor(
        trainer_type="single",
        model_types=["lightgbm", "xgboost"],
        training_service_cls=MockServiceClassMulti
    )
    
    try:
        prediction = predictor_multi.predict(df)
        logger.info(f"Multi-model prediction: {prediction}")
        
        expected_trend = (0.8 + 0.6) / 2
        if abs(prediction["trend"] - expected_trend) < 1e-6:
             logger.info("✅ Averaging correct.")
        else:
             logger.error(f"❌ Averaging incorrect. Expected {expected_trend}, got {prediction['trend']}")

    except Exception as e:
        logger.error(f"❌ Multi-model predict failed: {e}")

if __name__ == "__main__":
    verify_hybrid_predictor()
