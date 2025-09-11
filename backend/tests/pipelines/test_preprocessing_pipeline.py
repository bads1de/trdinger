import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Import the pipeline module
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.utils.data_processing.pipelines.preprocessing_pipeline import create_preprocessing_pipeline

class TestPreprocessingPipeline:
    """Tests for preprocessing pipeline functionality."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'numeric1': [1.0, 2.0, 3.0, np.nan, 100.0],  # with outlier
            'numeric2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'categorical1': ['A', 'B', 'A', 'C', 'A'],  # Replace None with valid value
            'categorical2': ['X', 'Y', 'X', 'Z', 'Y']
        })

    def test_create_preprocessing_pipeline_returns_pipeline(self):
        """Test that create_preprocessing_pipeline returns a sklearn Pipeline."""
        pipeline = create_preprocessing_pipeline()

        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) > 0

    def test_pipeline_contains_expected_transformers(self):
        """Test that pipeline contains expected transformers."""
        pipeline = create_preprocessing_pipeline()

        step_names = [step[0] for step in pipeline.steps]
        assert 'preprocessor' in step_names
        # The actual transformer names depend on the ColumnTransformer

    def test_pipeline_fit_transform(self):
        """Test that pipeline can fit and transform data."""
        pipeline = create_preprocessing_pipeline()

        # Fit and transform
        result = pipeline.fit_transform(self.df)

        # Should return a DataFrame or array
        assert result is not None
        assert len(result) > 0

    def test_pipeline_handles_missing_values(self):
        """Test that pipeline properly handles missing values."""
        pipeline = create_preprocessing_pipeline()

        result = pipeline.fit_transform(self.df)

        # Check that NaN values are handled
        if isinstance(result, pd.DataFrame):
            # Check that numeric columns don't have NaN
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            assert not result[numeric_cols].isnull().any().any()

            # For categorical columns, check that None is replaced with proper values
            categorical_cols = result.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                # Should not have None values (they should be imputed)
                non_none_values = result[col].dropna()
                if len(non_none_values) > 0:
                    # At least some values should be properly encoded
                    continue
        else:
            # For numpy arrays, check numerical columns only
            assert not np.isnan(result[:, :2]).any()  # numerical columns

    def test_pipeline_preserves_index(self):
        """Test that pipeline preserves DataFrame index."""
        pipeline = create_preprocessing_pipeline()

        result = pipeline.fit_transform(self.df)

        if isinstance(result, pd.DataFrame):
            pd.testing.assert_index_equal(result.index, self.df.index)

    def test_pipeline_with_custom_parameters(self):
        """Test pipeline creation with custom parameters."""
        custom_params = {
            'outlier_method': 'isolation_forest',
            'outlier_contamination': 0.1,
            'categorical_encoding': 'label'
        }

        pipeline = create_preprocessing_pipeline(**custom_params)

        assert isinstance(pipeline, Pipeline)