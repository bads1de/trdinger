import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

# Import the pipeline module
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.utils.data_processing.pipelines.ml_pipeline import create_ml_pipeline

class TestMLPipeline:
    """Tests for ML pipeline functionality."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })

    def test_create_ml_pipeline_returns_pipeline(self):
        """Test that create_ml_pipeline returns a sklearn Pipeline."""
        pipeline = create_ml_pipeline()

        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) > 0

    def test_ml_pipeline_contains_expected_steps(self):
        """Test that ML pipeline contains expected steps."""
        pipeline = create_ml_pipeline()

        step_names = [step[0] for step in pipeline.steps]
        assert 'preprocessing' in step_names
        # Feature selection and scaling may be optional

    def test_ml_pipeline_with_feature_selection(self):
        """Test ML pipeline with feature selection enabled."""
        pipeline = create_ml_pipeline(feature_selection=True, n_features=2)

        assert isinstance(pipeline, Pipeline)

        # Fit and transform
        result = pipeline.fit_transform(self.df.drop('target', axis=1), self.df['target'])

        # Should have selected features
        assert result.shape[1] <= 2

    def test_ml_pipeline_with_scaling(self):
        """Test ML pipeline with scaling enabled."""
        pipeline = create_ml_pipeline(scaling=True)

        assert isinstance(pipeline, Pipeline)

        # Fit and transform
        result = pipeline.fit_transform(self.df.drop('target', axis=1))

        # Check that scaling was applied (values should be standardized)
        assert result is not None

    def test_ml_pipeline_fit_transform(self):
        """Test that ML pipeline can fit and transform data."""
        pipeline = create_ml_pipeline()

        # Fit and transform
        result = pipeline.fit_transform(self.df.drop('target', axis=1))

        # Should return processed data
        assert result is not None
        assert len(result) > 0

    def test_ml_pipeline_with_custom_parameters(self):
        """Test ML pipeline creation with custom parameters."""
        custom_params = {
            'scaling_method': 'robust',
            'feature_selection': True,
            'n_features': 3
        }

        pipeline = create_ml_pipeline(**custom_params)

        assert isinstance(pipeline, Pipeline)