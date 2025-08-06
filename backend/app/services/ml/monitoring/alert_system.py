"""
自動アラートシステム

閾値超過時の通知、エスカレーション、自動対応を管理します。
"""

import logging
import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp

from .realtime_monitor import Alert, AlertLevel

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """通知チャネル"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    LOG = "log"
    CONSOLE = "console"


@dataclass
class NotificationConfig:
    """通知設定"""
    channel: NotificationChannel
    enabled: bool = True
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}


@dataclass
class EscalationRule:
    """エスカレーションルール"""
    alert_level: AlertLevel
    delay_minutes: int
    notification_channels: List[NotificationChannel]
    auto_actions: List[str] = None
    
    def __post_init__(self):
        if self.auto_actions is None:
            self.auto_actions = []


class AlertSystem:
    """
    自動アラートシステム
    
    アラートの通知、エスカレーション、自動対応を管理します。
    """
    
    def __init__(self):
        """初期化"""
        self.notification_configs: Dict[NotificationChannel, NotificationConfig] = {}
        self.escalation_rules: List[EscalationRule] = []
        self.pending_escalations: List[Dict[str, Any]] = []
        self.auto_actions: Dict[str, Callable] = {}
        
        # デフォルト設定
        self._setup_default_configs()
        self._setup_default_escalation_rules()
        self._setup_auto_actions()
    
    def _setup_default_configs(self):
        """デフォルト通知設定"""
        # ログ通知（常に有効）
        self.notification_configs[NotificationChannel.LOG] = NotificationConfig(
            channel=NotificationChannel.LOG,
            enabled=True
        )
        
        # コンソール通知
        self.notification_configs[NotificationChannel.CONSOLE] = NotificationConfig(
            channel=NotificationChannel.CONSOLE,
            enabled=True
        )
        
        # メール通知（設定が必要）
        self.notification_configs[NotificationChannel.EMAIL] = NotificationConfig(
            channel=NotificationChannel.EMAIL,
            enabled=False,
            config={
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "recipients": []
            }
        )
        
        # Webhook通知
        self.notification_configs[NotificationChannel.WEBHOOK] = NotificationConfig(
            channel=NotificationChannel.WEBHOOK,
            enabled=False,
            config={
                "url": "",
                "headers": {},
                "timeout": 30
            }
        )
    
    def _setup_default_escalation_rules(self):
        """デフォルトエスカレーションルール"""
        # 即座に通知
        self.escalation_rules.append(EscalationRule(
            alert_level=AlertLevel.CRITICAL,
            delay_minutes=0,
            notification_channels=[NotificationChannel.LOG, NotificationChannel.CONSOLE],
            auto_actions=["log_critical", "attempt_recovery"]
        ))
        
        self.escalation_rules.append(EscalationRule(
            alert_level=AlertLevel.ERROR,
            delay_minutes=0,
            notification_channels=[NotificationChannel.LOG, NotificationChannel.CONSOLE],
            auto_actions=["log_error"]
        ))
        
        self.escalation_rules.append(EscalationRule(
            alert_level=AlertLevel.WARNING,
            delay_minutes=5,
            notification_channels=[NotificationChannel.LOG],
            auto_actions=["log_warning"]
        ))
        
        # 5分後にエスカレーション
        self.escalation_rules.append(EscalationRule(
            alert_level=AlertLevel.CRITICAL,
            delay_minutes=5,
            notification_channels=[NotificationChannel.EMAIL, NotificationChannel.WEBHOOK],
            auto_actions=["escalate_to_admin"]
        ))
    
    def _setup_auto_actions(self):
        """自動対応アクションを設定"""
        self.auto_actions = {
            "log_critical": self._log_critical_action,
            "log_error": self._log_error_action,
            "log_warning": self._log_warning_action,
            "attempt_recovery": self._attempt_recovery_action,
            "escalate_to_admin": self._escalate_to_admin_action,
            "restart_service": self._restart_service_action,
            "clear_cache": self._clear_cache_action
        }
    
    def configure_notification(self, channel: NotificationChannel, config: Dict[str, Any]):
        """通知設定を更新"""
        if channel in self.notification_configs:
            self.notification_configs[channel].config.update(config)
            self.notification_configs[channel].enabled = True
            logger.info(f"通知設定を更新しました: {channel.value}")
        else:
            logger.error(f"未知の通知チャネル: {channel}")
    
    def add_escalation_rule(self, rule: EscalationRule):
        """エスカレーションルールを追加"""
        self.escalation_rules.append(rule)
        logger.info(f"エスカレーションルールを追加: {rule.alert_level.value}")
    
    async def handle_alert(self, alert: Alert):
        """アラートを処理"""
        try:
            logger.info(f"アラート処理開始: {alert.level.value} - {alert.message}")
            
            # 適用可能なエスカレーションルールを取得
            applicable_rules = [
                rule for rule in self.escalation_rules 
                if rule.alert_level == alert.level
            ]
            
            for rule in applicable_rules:
                if rule.delay_minutes == 0:
                    # 即座に実行
                    await self._execute_rule(alert, rule)
                else:
                    # 遅延実行をスケジュール
                    self._schedule_escalation(alert, rule)
            
        except Exception as e:
            logger.error(f"アラート処理エラー: {e}")
    
    async def _execute_rule(self, alert: Alert, rule: EscalationRule):
        """エスカレーションルールを実行"""
        try:
            # 通知送信
            for channel in rule.notification_channels:
                await self._send_notification(alert, channel)
            
            # 自動対応実行
            for action_name in rule.auto_actions:
                if action_name in self.auto_actions:
                    try:
                        await self.auto_actions[action_name](alert)
                    except Exception as e:
                        logger.error(f"自動対応エラー ({action_name}): {e}")
                else:
                    logger.warning(f"未知の自動対応: {action_name}")
                    
        except Exception as e:
            logger.error(f"エスカレーションルール実行エラー: {e}")
    
    def _schedule_escalation(self, alert: Alert, rule: EscalationRule):
        """エスカレーションをスケジュール"""
        escalation_time = datetime.now() + timedelta(minutes=rule.delay_minutes)
        
        self.pending_escalations.append({
            "alert": alert,
            "rule": rule,
            "scheduled_time": escalation_time
        })
        
        logger.info(f"エスカレーションをスケジュール: {rule.delay_minutes}分後")
    
    async def _send_notification(self, alert: Alert, channel: NotificationChannel):
        """通知を送信"""
        config = self.notification_configs.get(channel)
        if not config or not config.enabled:
            return
        
        try:
            if channel == NotificationChannel.LOG:
                await self._send_log_notification(alert)
            elif channel == NotificationChannel.CONSOLE:
                await self._send_console_notification(alert)
            elif channel == NotificationChannel.EMAIL:
                await self._send_email_notification(alert, config)
            elif channel == NotificationChannel.WEBHOOK:
                await self._send_webhook_notification(alert, config)
                
        except Exception as e:
            logger.error(f"通知送信エラー ({channel.value}): {e}")
    
    async def _send_log_notification(self, alert: Alert):
        """ログ通知"""
        log_level = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical
        }.get(alert.level, logger.info)
        
        log_level(f"[ALERT] {alert.category}: {alert.message}")
    
    async def _send_console_notification(self, alert: Alert):
        """コンソール通知"""
        timestamp = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        level_emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨"
        }.get(alert.level, "📢")
        
        print(f"{level_emoji} [{timestamp}] {alert.level.value.upper()}: {alert.message}")
    
    async def _send_email_notification(self, alert: Alert, config: NotificationConfig):
        """メール通知"""
        if not config.config.get("recipients"):
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = config.config.get("username", "")
            msg['To'] = ", ".join(config.config["recipients"])
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.category}: {alert.message}"

            body = self._format_alert_email(alert)
            msg.attach(MIMEText(body, 'html'))
            
            server = smtplib.SMTP(config.config["smtp_server"], config.config["smtp_port"])
            server.starttls()
            server.login(config.config["username"], config.config["password"])
            server.send_message(msg)
            server.quit()
            
            logger.info("メール通知を送信しました")
            
        except Exception as e:
            logger.error(f"メール送信エラー: {e}")
    
    async def _send_webhook_notification(self, alert: Alert, config: NotificationConfig):
        """Webhook通知"""
        if not config.config.get("url"):
            return
        
        try:
            payload = {
                "timestamp": alert.timestamp.isoformat(),
                "level": alert.level.value,
                "category": alert.category,
                "message": alert.message,
                "details": alert.details
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config.config["url"],
                    json=payload,
                    headers=config.config.get("headers", {}),
                    timeout=config.config.get("timeout", 30)
                ) as response:
                    if response.status == 200:
                        logger.info("Webhook通知を送信しました")
                    else:
                        logger.error(f"Webhook送信失敗: {response.status}")
                        
        except Exception as e:
            logger.error(f"Webhook送信エラー: {e}")
    
    def _format_alert_email(self, alert: Alert) -> str:
        """アラートメールをフォーマット"""
        return f"""
        <html>
        <body>
            <h2>システムアラート</h2>
            <table border="1" cellpadding="5">
                <tr><td><b>時刻</b></td><td>{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
                <tr><td><b>レベル</b></td><td>{alert.level.value.upper()}</td></tr>
                <tr><td><b>カテゴリ</b></td><td>{alert.category}</td></tr>
                <tr><td><b>メッセージ</b></td><td>{alert.message}</td></tr>
            </table>
            
            <h3>詳細情報</h3>
            <pre>{json.dumps(alert.details, indent=2, ensure_ascii=False)}</pre>
        </body>
        </html>
        """
    
    # 自動対応アクション
    async def _log_critical_action(self, alert: Alert):
        """クリティカルログアクション"""
        logger.critical(f"CRITICAL ALERT: {alert.message} | Details: {alert.details}")
    
    async def _log_error_action(self, alert: Alert):
        """エラーログアクション"""
        logger.error(f"ERROR ALERT: {alert.message}")
    
    async def _log_warning_action(self, alert: Alert):
        """警告ログアクション"""
        logger.warning(f"WARNING ALERT: {alert.message}")
    
    async def _attempt_recovery_action(self, alert: Alert):
        """復旧試行アクション"""
        logger.info(f"復旧を試行中: {alert.category}")
        # 実際の復旧ロジックをここに実装
    
    async def _escalate_to_admin_action(self, alert: Alert):
        """管理者エスカレーションアクション"""
        logger.critical(f"管理者にエスカレーション: {alert.message}")
    
    async def _restart_service_action(self, alert: Alert):
        """サービス再起動アクション"""
        logger.warning("サービス再起動が要求されました（実装待ち）")
    
    async def _clear_cache_action(self, alert: Alert):
        """キャッシュクリアアクション"""
        logger.info("キャッシュクリアが要求されました（実装待ち）")
    
    async def process_pending_escalations(self):
        """保留中のエスカレーションを処理"""
        current_time = datetime.now()
        
        # 実行すべきエスカレーションを特定
        to_execute = [
            escalation for escalation in self.pending_escalations
            if escalation["scheduled_time"] <= current_time
        ]
        
        # 実行
        for escalation in to_execute:
            try:
                await self._execute_rule(escalation["alert"], escalation["rule"])
                self.pending_escalations.remove(escalation)
            except Exception as e:
                logger.error(f"エスカレーション実行エラー: {e}")
    
    def get_alert_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """アラート統計を取得"""
        # 実装は簡略化（実際は永続化されたアラート履歴から取得）
        return {
            "period_hours": hours,
            "pending_escalations": len(self.pending_escalations),
            "enabled_channels": [
                channel.value for channel, config in self.notification_configs.items()
                if config.enabled
            ],
            "escalation_rules_count": len(self.escalation_rules)
        }
