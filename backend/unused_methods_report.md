# 未使用メソッド・関数レポート (60%信頼度)

## API 関数 (FastAPI エンドポイント)

### Auto Strategy API

- `app/api/auto_strategy.py:117` - `generate_strategy()`
- `app/api/auto_strategy.py:216` - `get_experiment_results()`
- `app/api/auto_strategy.py:284` - `get_config_presets()`

### Data Collection API

- `app/api/data_collection.py:65` - `update_bulk_incremental_data()`
- `app/api/data_collection.py:90` - `collect_bitcoin_full_data()`
- `app/api/data_collection.py:117` - `collect_bulk_historical_data()`
- `app/api/data_collection.py:197` - `collect_all_data_bulk()`

### Dependencies API

- `app/api/dependencies.py:29` - `get_backtest_service()`
- `app/api/dependencies.py:50` - `get_backtest_service_with_db()`
- `app/api/dependencies.py:71` - `get_strategy_integration_service()`
- `app/api/dependencies.py:84` - `get_market_data_orchestration_service()`

### Funding Rates API

- `app/api/funding_rates.py:24` - `get_funding_rates()`
- `app/api/funding_rates.py:118` - `bulk_collect_funding_rates()`

### ML Management API

- `app/api/ml_management.py:29` - `get_models()`
- `app/api/ml_management.py:72` - `delete_all_models_legacy()`
- `app/api/ml_management.py:190` - `get_current_model()`
- `app/api/ml_management.py:227` - `get_ml_config()`

### ML Training API

- `app/api/ml_training.py:194` - `start_ml_training()`
- `app/api/ml_training.py:248` - `get_ml_training_status()`
- `app/api/ml_training.py:259` - `get_ml_model_info()`
- `app/api/ml_training.py:273` - `stop_ml_training()`

### Utils

- `app/services/auto_strategy/utils/auto_strategy_utils.py:194` - `denormalize_parameter()`
- `app/services/auto_strategy/utils/auto_strategy_utils.py:322` - `extract_config_subset()`
- `app/services/auto_strategy/utils/common_utils.py:71` - `percentage_to_decimal()`
- `app/services/auto_strategy/utils/common_utils.py:95` - `log_function_entry()`
- `app/services/auto_strategy/utils/common_utils.py:102` - `log_function_exit()`
- `app/services/auto_strategy/utils/common_utils.py:117` - `log_business_event()`
- `app/services/auto_strategy/utils/common_utils.py:156` - `validate_positive_number()`
- `app/services/auto_strategy/utils/common_utils.py:164` - `validate_list_not_empty()`
- `app/services/auto_strategy/utils/common_utils.py:197` - `time_async_function()`
- `app/services/auto_strategy/utils/common_utils.py:244` - `cleanup_expired()`
- `app/services/auto_strategy/utils/data_coverage_analyzer.py:36` - `analyze_strategy_coverage()`
- `app/services/auto_strategy/utils/data_coverage_analyzer.py:226` - `get_coverage_summary()`
- `app/services/auto_strategy/utils/error_handling.py:87` - `handle_validation_error()`
- `app/services/auto_strategy/utils/error_handling.py:170` - `retry_on_failure()`
- `app/services/auto_strategy/utils/error_handling.py:213` - `validate_and_execute()`
- `app/services/auto_strategy/utils/error_handling.py:269` - `add_error()`
- `app/services/auto_strategy/utils/error_handling.py:274` - `add_warning()`
- `app/services/auto_strategy/utils/error_handling.py:287` - `get_summary()`
- `app/services/auto_strategy/utils/error_handling.py:304` - `error_boundary()`
- `app/services/auto_strategy/utils/operand_grouping.py:223` - `get_operands_by_group()`
- `app/services/auto_strategy/utils/operand_grouping.py:238` - `get_group_statistics()`
- `app/services/auto_strategy/utils/strategy_integration_service.py:86` - `get_all_strategies_for_stats()`

### Technical Indicators

- `app/services/indicators/technical_indicators/math_operators.py:57` - `div()`
- `app/services/indicators/technical_indicators/math_operators.py:94` - `sum_values()`
- `app/services/indicators/technical_indicators/math_transform.py:27` - `acos()`
- `app/services/indicators/technical_indicators/math_transform.py:50` - `asin()`
- `app/services/indicators/technical_indicators/math_transform.py:64` - `atan()`
- `app/services/indicators/technical_indicators/math_transform.py:100` - `ln()`

### Pattern Recognition

- `app/services/indicators/technical_indicators/pattern_recognition.py:82` - `cdl_hanging_man()`
- `app/services/indicators/technical_indicators/pattern_recognition.py:167` - `cdl_harami()`
- `app/services/indicators/technical_indicators/pattern_recognition.py:195` - `cdl_piercing()`
- `app/services/indicators/technical_indicators/pattern_recognition.py:223` - `cdl_dark_cloud_cover()`
- `app/services/indicators/technical_indicators/pattern_recognition.py:309` - `cdl_three_black_crows()`
- `app/services/indicators/technical_indicators/pattern_recognition.py:337` - `cdl_three_white_soldiers()`
- `app/services/indicators/technical_indicators/pattern_recognition.py:366` - `cdl_marubozu()`
- `app/services/indicators/technical_indicators/pattern_recognition.py:394` - `cdl_spinning_top()`
- `app/services/indicators/technical_indicators/pattern_recognition.py:426` - `doji()`
- `app/services/indicators/technical_indicators/pattern_recognition.py:431` - `hammer()`
- `app/services/indicators/technical_indicators/pattern_recognition.py:436` - `engulfing_pattern()`
- `app/services/indicators/technical_indicators/pattern_recognition.py:441` - `morning_star()`
- `app/services/indicators/technical_indicators/pattern_recognition.py:446` - `evening_star()`

## ML 関連

### Feature Engineering

- `app/services/ml/feature_engineering/advanced_features.py:402` - `clean_features()`
- `app/services/ml/feature_engineering/automl_features/autofeat_calculator.py:458` - `_extract_selected_features()`
- `app/services/ml/feature_engineering/automl_features/autofeat_calculator.py:504` - `_calculate_feature_scores()`
- `app/services/ml/feature_engineering/automl_features/autofeat_calculator.py:559` - `get_generation_info()`
- `app/services/ml/feature_engineering/automl_features/autofeat_calculator.py:563` - `get_feature_scores()`
- `app/services/ml/feature_engineering/automl_features/autofeat_calculator.py:567` - `evaluate_selected_features()`
- `app/services/ml/feature_engineering/automl_features/feature_selector.py:386` - `get_selection_summary()`
- `app/services/ml/feature_engineering/automl_features/feature_selector.py:400` - `get_feature_scores()`
- `app/services/ml/feature_engineering/automl_features/feature_selector.py:404` - `clear_history()`
- `app/services/ml/feature_engineering/automl_features/feature_settings.py:238` - `get_profile()`
- `app/services/ml/feature_engineering/automl_features/feature_settings.py:242` - `get_profiles_by_category()`
- `app/services/ml/feature_engineering/automl_features/feature_settings.py:299` - `save_settings_to_file()`
- `app/services/ml/feature_engineering/automl_features/feature_settings.py:304` - `load_settings_from_file()`
- `app/services/ml/feature_engineering/automl_features/feature_settings.py:309` - `get_all_profile_names()`
- `app/services/ml/feature_engineering/automl_features/feature_settings.py:313` - `get_profile_summary()`
- `app/services/ml/feature_engineering/automl_features/memory_utils.py:42` - `calculate_optimal_batch_size()`
- `app/services/ml/feature_engineering/automl_features/memory_utils.py:76` - `optimize_dataframe_dtypes()`
- `app/services/ml/feature_engineering/automl_features/memory_utils.py:166` - `memory_monitor_decorator()`
- `app/services/ml/feature_engineering/automl_features/memory_utils.py:185` - `check_memory_availability()`
- `app/services/ml/feature_engineering/automl_features/memory_utils.py:233` - `get_memory_efficient_autofeat_config()`
- `app/services/ml/feature_engineering/automl_features/memory_utils.py:294` - `cleanup_autofeat_memory()`
- `app/services/ml/feature_engineering/automl_features/memory_utils.py:320` - `log_memory_usage()`
- `app/services/ml/feature_engineering/automl_features/performance_optimizer.py:85` - `get_cached_features()`
- `app/services/ml/feature_engineering/automl_features/performance_optimizer.py:93` - `cache_features()`
- `app/services/ml/feature_engineering/automl_features/performance_optimizer.py:100` - `_cleanup_cache_if_needed()`
- `app/services/ml/feature_engineering/automl_features/performance_optimizer.py:266` - `get_cache_stats()`
- `app/services/ml/feature_engineering/automl_features/performance_optimizer.py:306` - `cleanup_autofeat_memory()`
- `app/services/ml/feature_engineering/automl_features/performance_optimizer.py:406` - `get_memory_recommendations()`
- `app/services/ml/feature_engineering/automl_features/performance_optimizer.py:550` - `memory_profiling_decorator()`
- `app/services/ml/feature_engineering/automl_features/tsfresh_calculator.py:489` - `get_extraction_info()`
- `app/services/ml/feature_engineering/automl_features/tsfresh_calculator.py:615` - `set_market_regime()`
- `app/services/ml/feature_engineering/automl_features/tsfresh_calculator.py:620` - `get_regime_info()`
- `app/services/ml/feature_engineering/base_feature_calculator.py:128` - `log_feature_calculation_start()`
- `app/services/ml/feature_engineering/data_frequency_manager.py:46` - `get_target_frequency()`
- `app/services/ml/feature_engineering/data_frequency_manager.py:315` - `validate_data_consistency()`
- `app/services/ml/feature_engineering/enhanced_crypto_features.py:36` - `create_comprehensive_features()`
- `app/services/ml/feature_engineering/enhanced_crypto_features.py:426` - `get_feature_groups()`
- `app/services/ml/feature_engineering/enhanced_crypto_features.py:430` - `get_top_features_by_correlation()`
- `app/services/ml/feature_engineering/feature_engineering_service.py:955` - `get_cache_info()`
- `app/services/ml/feature_engineering/feature_engineering_service.py:1043` - `get_automl_config()`
- `app/services/ml/feature_engineering/feature_engineering_service.py:1049` - `set_automl_config()`
- `app/services/ml/feature_engineering/optimized_crypto_features.py:36` - `create_optimized_features()`
- `app/services/ml/feature_engineering/optimized_crypto_features.py:599` - `get_feature_groups()`
- `app/services/ml/feature_engineering/optimized_crypto_features.py:603` - `get_top_features_by_stability()`

## 統計情報

- **総未使用メソッド・関数数**: 約 200 個
- **最も多いカテゴリ**: ML 関連 (約 80 個)
- **API 関数**: 約 25 個
- **データベースリポジトリ**: 約 30 個
- **Auto Strategy 関連**: 約 50 個

## 推奨事項

1. **高優先度**: API エンドポイントの未使用関数は削除を検討
2. **中優先度**: ユーティリティ関数やヘルパーメソッドは将来使用される可能性があるため慎重に判断
3. **低優先度**: テスト関連のメソッドは開発中に使用される可能性があるため保持を推奨

注意: 60%信頼度のため、実際の使用状況を確認してから削除することを強く推奨します。
