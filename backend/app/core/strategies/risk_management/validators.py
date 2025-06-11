"""
リスク管理パラメータ検証

リスク管理設定の妥当性を検証する機能
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def validate_risk_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    リスク管理パラメータの妥当性を検証
    
    Args:
        config: リスク管理設定辞書
        
    Returns:
        検証済み・正規化された設定辞書
        
    Raises:
        ValueError: パラメータが無効な場合
        
    Example:
        >>> config = {
        ...     "stop_loss": {"type": "percentage", "value": 0.02},
        ...     "take_profit": {"type": "percentage", "value": 0.05}
        ... }
        >>> validated = validate_risk_parameters(config)
    """
    errors = []
    validated_config = {}
    
    try:
        # ストップロス設定の検証
        if "stop_loss" in config:
            sl_config = config["stop_loss"]
            validated_sl = _validate_sl_tp_config(sl_config, "stop_loss")
            if validated_sl:
                validated_config["stop_loss"] = validated_sl
            else:
                errors.append("Invalid stop_loss configuration")
        
        # テイクプロフィット設定の検証
        if "take_profit" in config:
            tp_config = config["take_profit"]
            validated_tp = _validate_sl_tp_config(tp_config, "take_profit")
            if validated_tp:
                validated_config["take_profit"] = validated_tp
            else:
                errors.append("Invalid take_profit configuration")
        
        # トレーリングストップ設定の検証
        if "trailing_stop" in config:
            trailing_config = config["trailing_stop"]
            validated_trailing = _validate_trailing_config(trailing_config)
            if validated_trailing:
                validated_config["trailing_stop"] = validated_trailing
            else:
                errors.append("Invalid trailing_stop configuration")
        
        # ATRベース設定の検証
        if "atr_based" in config:
            atr_config = config["atr_based"]
            validated_atr = _validate_atr_config(atr_config)
            if validated_atr:
                validated_config["atr_based"] = validated_atr
            else:
                errors.append("Invalid atr_based configuration")
        
        if errors:
            raise ValueError(f"Validation errors: {', '.join(errors)}")
            
        return validated_config
        
    except Exception as e:
        logger.error(f"Error validating risk parameters: {e}")
        raise


def _validate_sl_tp_config(config: Dict[str, Any], config_type: str) -> Optional[Dict[str, Any]]:
    """
    ストップロス・テイクプロフィット設定の検証
    
    Args:
        config: SL/TP設定辞書
        config_type: 設定タイプ（"stop_loss" or "take_profit"）
        
    Returns:
        検証済み設定辞書（None if 無効）
    """
    try:
        if not isinstance(config, dict):
            logger.error(f"{config_type} must be a dictionary")
            return None
        
        # 必須フィールドの確認
        if "type" not in config:
            logger.error(f"{config_type} must have 'type' field")
            return None
        
        if "value" not in config:
            logger.error(f"{config_type} must have 'value' field")
            return None
        
        config_type_value = config["type"]
        value = config["value"]
        
        # タイプの検証
        valid_types = ["percentage", "absolute", "atr"]
        if config_type_value not in valid_types:
            logger.error(f"Invalid {config_type} type: {config_type_value}. Must be one of {valid_types}")
            return None
        
        # 値の検証
        if config_type_value == "percentage":
            if not isinstance(value, (int, float)) or value <= 0 or value >= 1:
                logger.error(f"Percentage {config_type} value must be between 0 and 1")
                return None
        
        elif config_type_value == "absolute":
            if not isinstance(value, (int, float)) or value <= 0:
                logger.error(f"Absolute {config_type} value must be positive")
                return None
        
        elif config_type_value == "atr":
            if not isinstance(value, (int, float)) or value <= 0:
                logger.error(f"ATR {config_type} multiplier must be positive")
                return None
        
        return {
            "type": config_type_value,
            "value": float(value)
        }
        
    except Exception as e:
        logger.error(f"Error validating {config_type} config: {e}")
        return None


def _validate_trailing_config(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    トレーリングストップ設定の検証
    
    Args:
        config: トレーリングストップ設定辞書
        
    Returns:
        検証済み設定辞書（None if 無効）
    """
    try:
        if not isinstance(config, dict):
            logger.error("trailing_stop must be a dictionary")
            return None
        
        # 必須フィールド
        if "enabled" not in config:
            config["enabled"] = True
        
        if not isinstance(config["enabled"], bool):
            logger.error("trailing_stop.enabled must be boolean")
            return None
        
        if not config["enabled"]:
            return {"enabled": False}
        
        # トレーリング方式の検証
        method = config.get("method", "percentage")
        valid_methods = ["percentage", "atr", "absolute"]
        
        if method not in valid_methods:
            logger.error(f"Invalid trailing method: {method}. Must be one of {valid_methods}")
            return None
        
        # 値の検証
        if "value" not in config:
            logger.error("trailing_stop must have 'value' field when enabled")
            return None
        
        value = config["value"]
        if not isinstance(value, (int, float)) or value <= 0:
            logger.error("trailing_stop value must be positive")
            return None
        
        return {
            "enabled": True,
            "method": method,
            "value": float(value)
        }
        
    except Exception as e:
        logger.error(f"Error validating trailing config: {e}")
        return None


def _validate_atr_config(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    ATRベース設定の検証
    
    Args:
        config: ATRベース設定辞書
        
    Returns:
        検証済み設定辞書（None if 無効）
    """
    try:
        if not isinstance(config, dict):
            logger.error("atr_based must be a dictionary")
            return None
        
        # ATR期間の検証
        period = config.get("period", 14)
        if not isinstance(period, int) or period <= 0:
            logger.error("ATR period must be positive integer")
            return None
        
        # ストップロス倍数の検証
        sl_multiplier = config.get("sl_multiplier", 2.0)
        if not isinstance(sl_multiplier, (int, float)) or sl_multiplier <= 0:
            logger.error("ATR stop loss multiplier must be positive")
            return None
        
        # テイクプロフィット倍数の検証
        tp_multiplier = config.get("tp_multiplier", 3.0)
        if not isinstance(tp_multiplier, (int, float)) or tp_multiplier <= 0:
            logger.error("ATR take profit multiplier must be positive")
            return None
        
        return {
            "period": int(period),
            "sl_multiplier": float(sl_multiplier),
            "tp_multiplier": float(tp_multiplier)
        }
        
    except Exception as e:
        logger.error(f"Error validating ATR config: {e}")
        return None


def validate_risk_reward_ratio(
    entry_price: float,
    sl_price: float,
    tp_price: float,
    min_ratio: float = 1.0,
    is_long: bool = True
) -> bool:
    """
    リスクリワード比率の妥当性を検証
    
    Args:
        entry_price: エントリー価格
        sl_price: ストップロス価格
        tp_price: テイクプロフィット価格
        min_ratio: 最小リスクリワード比率
        is_long: ロングポジションかどうか
        
    Returns:
        比率が最小値以上の場合True
    """
    try:
        if is_long:
            risk = entry_price - sl_price
            reward = tp_price - entry_price
        else:
            risk = sl_price - entry_price
            reward = entry_price - tp_price
        
        if risk <= 0:
            logger.warning(f"Invalid risk value: {risk}")
            return False
        
        ratio = reward / risk
        
        if ratio < min_ratio:
            logger.warning(f"Risk-reward ratio {ratio:.2f} below minimum {min_ratio}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating risk-reward ratio: {e}")
        return False
