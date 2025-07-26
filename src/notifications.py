#!/usr/bin/env python3
"""
SHIOL+ NotificationEngine Component
===================================

NotificationEngine class for email alerts and error handling with SMTP integration.
Provides comprehensive notification capabilities for pipeline execution results,
system health alerts, error notifications, and report delivery.

Features:
- SMTP email integration with TLS/SSL support
- Multiple notification levels (success, warning, error)
- HTML and plain text email templates
- Recipient management and filtering
- Rate limiting to prevent spam
- Integration with pipeline execution results
- Retry logic for failed email delivery
- Professional SHIOL+ branding
"""

import configparser
import os
import smtplib
import ssl
import time
import json
import hashlib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from collections import defaultdict, deque

# Logging
from loguru import logger


class NotificationEngine:
    """
    NotificationEngine class for comprehensive email notifications and alerts.
    
    Provides SMTP email integration with configuration from config.ini,
    multiple notification levels, HTML/plain text templates, rate limiting,
    and integration with pipeline execution results.
    """
    
    def __init__(self, config_path: str = "config/config.ini"):
        """
        Initialize the NotificationEngine.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_configuration()
        
        # Load notification configuration
        self._load_notification_config()
        
        # Rate limiting tracking
        self.rate_limit_tracker = defaultdict(deque)
        self.max_emails_per_hour = 50
        self.max_emails_per_day = 200
        
        # Retry configuration
        self.max_retry_attempts = 3
        self.retry_delay_seconds = 30
        self.retry_backoff_multiplier = 2.0
        
        # Notification history
        self.notification_history = deque(maxlen=100)
        
        logger.info("NotificationEngine initialized successfully")
    
    def _load_configuration(self) -> configparser.ConfigParser:
        """Load configuration from config.ini file."""
        try:
            config = configparser.ConfigParser()
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            config.read(self.config_path)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _load_notification_config(self):
        """Load notification-specific configuration from config.ini."""
        try:
            pipeline_section = self.config['pipeline']
            
            # SMTP settings
            self.smtp_server = pipeline_section.get('smtp_server', 'smtp.gmail.com')
            self.smtp_port = pipeline_section.getint('smtp_port', 587)
            self.smtp_use_tls = pipeline_section.getboolean('smtp_use_tls', True)
            self.smtp_use_ssl = pipeline_section.getboolean('smtp_use_ssl', False)
            self.smtp_username = pipeline_section.get('smtp_username', '')
            self.smtp_password = pipeline_section.get('smtp_password', '')
            
            # Notification settings
            recipients_str = pipeline_section.get('notification_recipients', 'admin@example.com')
            self.notification_recipients = [email.strip() for email in recipients_str.split(',') if email.strip()]
            
            levels_str = pipeline_section.get('notification_levels', 'error,warning')
            self.notification_levels = [level.strip().lower() for level in levels_str.split(',') if level.strip()]
            
            # Report settings
            self.generate_reports = pipeline_section.getboolean('generate_reports', True)
            report_formats_str = pipeline_section.get('report_format', 'html,csv')
            self.report_formats = [fmt.strip().lower() for fmt in report_formats_str.split(',') if fmt.strip()]
            self.report_retention_days = pipeline_section.getint('report_retention_days', 30)
            
            # Rate limiting settings
            self.max_emails_per_hour = pipeline_section.getint('max_emails_per_hour', 50)
            self.max_emails_per_day = pipeline_section.getint('max_emails_per_day', 200)
            
            logger.info(f"Notification config loaded: SMTP={self.smtp_server}:{self.smtp_port}, Recipients={len(self.notification_recipients)}")
            
        except Exception as e:
            logger.error(f"Failed to load notification configuration: {e}")
            # Set safe defaults
            self.smtp_server = 'smtp.gmail.com'
            self.smtp_port = 587
            self.smtp_use_tls = True
            self.smtp_use_ssl = False
            self.smtp_username = ''
            self.smtp_password = ''
            self.notification_recipients = []
            self.notification_levels = ['error', 'warning']
            self.generate_reports = True
            self.report_formats = ['html', 'csv']
            self.report_retention_days = 30
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate notification configuration.
        
        Returns:
            Dict with validation results
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'smtp_configured': False,
            'recipients_configured': False,
            'levels_configured': False
        }
        
        try:
            # Check SMTP configuration
            if not self.smtp_server:
                validation_results['errors'].append("SMTP server not configured")
                validation_results['valid'] = False
            else:
                validation_results['smtp_configured'] = True
            
            if not self.smtp_username or not self.smtp_password:
                validation_results['warnings'].append("SMTP credentials not configured - authentication may fail")
            
            # Check recipients
            if not self.notification_recipients:
                validation_results['errors'].append("No notification recipients configured")
                validation_results['valid'] = False
            else:
                validation_results['recipients_configured'] = True
                
                # Validate email formats
                import re
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                invalid_emails = [email for email in self.notification_recipients 
                                if not re.match(email_pattern, email)]
                if invalid_emails:
                    validation_results['errors'].append(f"Invalid email addresses: {invalid_emails}")
                    validation_results['valid'] = False
            
            # Check notification levels
            valid_levels = ['success', 'warning', 'error', 'all']
            invalid_levels = [level for level in self.notification_levels if level not in valid_levels]
            if invalid_levels:
                validation_results['warnings'].append(f"Invalid notification levels: {invalid_levels}")
            else:
                validation_results['levels_configured'] = True
            
            # Check port range
            if not (1 <= self.smtp_port <= 65535):
                validation_results['errors'].append(f"Invalid SMTP port: {self.smtp_port}")
                validation_results['valid'] = False
            
            logger.info(f"Configuration validation completed: {'Valid' if validation_results['valid'] else 'Invalid'}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error during configuration validation: {e}")
            validation_results['valid'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")
            return validation_results
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test SMTP connectivity and authentication.
        
        Returns:
            Dict with connection test results
        """
        test_results = {
            'success': False,
            'connection_established': False,
            'authentication_successful': False,
            'error': None,
            'server_info': None,
            'test_timestamp': datetime.now().isoformat()
        }
        
        try:
            logger.info(f"Testing SMTP connection to {self.smtp_server}:{self.smtp_port}")
            
            # Create SMTP connection
            if self.smtp_use_ssl:
                context = ssl.create_default_context()
                server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context)
            else:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                if self.smtp_use_tls:
                    context = ssl.create_default_context()
                    server.starttls(context=context)
            
            test_results['connection_established'] = True
            test_results['server_info'] = server.ehlo()[1].decode('utf-8') if server.ehlo()[0] == 250 else "Unknown"
            
            # Test authentication if credentials provided
            if self.smtp_username and self.smtp_password:
                server.login(self.smtp_username, self.smtp_password)
                test_results['authentication_successful'] = True
                logger.info("SMTP authentication successful")
            else:
                logger.warning("No SMTP credentials provided - skipping authentication test")
            
            server.quit()
            test_results['success'] = True
            logger.info("SMTP connection test successful")
            
        except smtplib.SMTPAuthenticationError as e:
            error_msg = f"SMTP authentication failed: {str(e)}"
            test_results['error'] = error_msg
            logger.error(error_msg)
        except smtplib.SMTPConnectError as e:
            error_msg = f"SMTP connection failed: {str(e)}"
            test_results['error'] = error_msg
            logger.error(error_msg)
        except Exception as e:
            error_msg = f"SMTP test failed: {str(e)}"
            test_results['error'] = error_msg
            logger.error(error_msg)
        
        return test_results
    
    def _check_rate_limits(self, recipient: str) -> bool:
        """
        Check if sending to recipient would exceed rate limits.
        
        Args:
            recipient: Email address to check
            
        Returns:
            True if within limits, False if would exceed
        """
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        # Clean old entries
        recipient_history = self.rate_limit_tracker[recipient]
        while recipient_history and recipient_history[0] < day_ago:
            recipient_history.popleft()
        
        # Count recent emails
        emails_last_hour = sum(1 for timestamp in recipient_history if timestamp > hour_ago)
        emails_last_day = len(recipient_history)
        
        # Check limits
        if emails_last_hour >= self.max_emails_per_hour:
            logger.warning(f"Rate limit exceeded for {recipient}: {emails_last_hour} emails in last hour")
            return False
        
        if emails_last_day >= self.max_emails_per_day:
            logger.warning(f"Rate limit exceeded for {recipient}: {emails_last_day} emails in last day")
            return False
        
        return True
    
    def _record_email_sent(self, recipient: str):
        """Record that an email was sent to recipient."""
        self.rate_limit_tracker[recipient].append(datetime.now())
    
    def _filter_recipients(self, recipients: Optional[List[str]] = None, 
                          notification_level: str = 'info') -> List[str]:
        """
        Filter recipients based on configuration and rate limits.
        
        Args:
            recipients: Optional list of specific recipients
            notification_level: Level of notification
            
        Returns:
            List of valid recipients
        """
        # Use provided recipients or default
        target_recipients = recipients if recipients else self.notification_recipients
        
        # Filter by notification level
        if notification_level.lower() not in self.notification_levels and 'all' not in self.notification_levels:
            logger.info(f"Notification level '{notification_level}' not enabled - skipping")
            return []
        
        # Filter by rate limits
        valid_recipients = []
        for recipient in target_recipients:
            if self._check_rate_limits(recipient):
                valid_recipients.append(recipient)
            else:
                logger.warning(f"Skipping {recipient} due to rate limits")
        
        return valid_recipients
    
    def _generate_html_template(self, template_type: str, **kwargs) -> str:
        """
        Generate HTML email template.
        
        Args:
            template_type: Type of template (pipeline, error, health, report)
            **kwargs: Template variables
            
        Returns:
            HTML template string
        """
        # Common HTML header with SHIOL+ branding
        html_header = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>SHIOL+ Notification</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 20px; background-color: #f4f4f4; }
                .container { max-width: 600px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px 10px 0 0; text-align: center; }
                .header h1 { margin: 0; font-size: 24px; }
                .content { padding: 20px; }
                .status-success { color: #28a745; font-weight: bold; }
                .status-warning { color: #ffc107; font-weight: bold; }
                .status-error { color: #dc3545; font-weight: bold; }
                .metrics { background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0; }
                .metrics table { width: 100%; border-collapse: collapse; }
                .metrics td { padding: 8px; border-bottom: 1px solid #dee2e6; }
                .metrics td:first-child { font-weight: bold; width: 40%; }
                .footer { text-align: center; padding: 20px; color: #6c757d; font-size: 12px; border-top: 1px solid #dee2e6; margin-top: 20px; }
                .timestamp { color: #6c757d; font-size: 12px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 SHIOL+ Notification</h1>
                    <p>Advanced Powerball Prediction System</p>
                </div>
                <div class="content">
        """
        
        html_footer = """
                </div>
                <div class="footer">
                    <p>This is an automated notification from SHIOL+ Powerball Prediction System</p>
                    <p class="timestamp">Generated: {timestamp}</p>
                </div>
            </div>
        </body>
        </html>
        """.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))
        
        # Template-specific content
        if template_type == 'pipeline':
            content = self._generate_pipeline_html_content(**kwargs)
        elif template_type == 'error':
            content = self._generate_error_html_content(**kwargs)
        elif template_type == 'health':
            content = self._generate_health_html_content(**kwargs)
        elif template_type == 'report':
            content = self._generate_report_html_content(**kwargs)
        else:
            content = f"<h2>Notification</h2><p>{kwargs.get('message', 'No message provided')}</p>"
        
        return html_header + content + html_footer
    
    def _generate_pipeline_html_content(self, **kwargs) -> str:
        """Generate HTML content for pipeline notifications."""
        status = kwargs.get('status', 'unknown')
        execution_time = kwargs.get('execution_time', 'unknown')
        results = kwargs.get('results', {})
        summary = kwargs.get('summary', {})
        
        status_class = f"status-{status}" if status in ['success', 'warning', 'error'] else "status-info"
        status_icon = "✅" if status == 'success' else "⚠️" if status == 'warning' else "❌" if status == 'error' else "ℹ️"
        
        content = f"""
        <h2>{status_icon} Pipeline Execution Report</h2>
        <p>Pipeline execution completed with status: <span class="{status_class}">{status.upper()}</span></p>
        
        <div class="metrics">
            <h3>Execution Summary</h3>
            <table>
                <tr><td>Status</td><td class="{status_class}">{status.upper()}</td></tr>
                <tr><td>Execution Time</td><td>{execution_time}</td></tr>
                <tr><td>Steps Completed</td><td>{summary.get('successful_steps', 0)}/{summary.get('total_steps', 0)}</td></tr>
                <tr><td>Success Rate</td><td>{summary.get('success_rate', '0%')}</td></tr>
                <tr><td>Pipeline Health</td><td>{summary.get('pipeline_health', 'unknown')}</td></tr>
            </table>
        </div>
        """
        
        if results:
            content += "<div class='metrics'><h3>Step Results</h3><table>"
            for step_name, step_result in results.items():
                step_status = step_result.get('status', 'unknown')
                step_icon = "✅" if step_status == 'success' else "❌"
                content += f"<tr><td>{step_name.replace('_', ' ').title()}</td><td>{step_icon} {step_status}</td></tr>"
            content += "</table></div>"
        
        return content
    
    def _generate_error_html_content(self, **kwargs) -> str:
        """Generate HTML content for error notifications."""
        error_type = kwargs.get('error_type', 'Unknown Error')
        error_message = kwargs.get('error_message', 'No error message provided')
        context = kwargs.get('context', {})
        traceback_info = kwargs.get('traceback', '')
        
        content = f"""
        <h2>🚨 Error Alert</h2>
        <p><strong>Error Type:</strong> {error_type}</p>
        <p><strong>Error Message:</strong> {error_message}</p>
        """
        
        if context:
            content += "<div class='metrics'><h3>Error Context</h3><table>"
            for key, value in context.items():
                content += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"
            content += "</table></div>"
        
        if traceback_info:
            content += f"""
            <div class="metrics">
                <h3>Technical Details</h3>
                <pre style="background: #f8f9fa; padding: 10px; border-radius: 5px; overflow-x: auto; font-size: 11px;">{traceback_info}</pre>
            </div>
            """
        
        return content
    
    def _generate_health_html_content(self, **kwargs) -> str:
        """Generate HTML content for health notifications."""
        health_status = kwargs.get('health_status', 'unknown')
        metrics = kwargs.get('metrics', {})
        alerts = kwargs.get('alerts', [])
        
        status_class = f"status-{health_status}" if health_status in ['healthy', 'warning', 'critical'] else "status-info"
        status_icon = "💚" if health_status == 'healthy' else "⚠️" if health_status == 'warning' else "🔴"
        
        content = f"""
        <h2>{status_icon} System Health Alert</h2>
        <p>System health status: <span class="{status_class}">{health_status.upper()}</span></p>
        """
        
        if metrics:
            content += "<div class='metrics'><h3>System Metrics</h3><table>"
            for metric, value in metrics.items():
                content += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{value}</td></tr>"
            content += "</table></div>"
        
        if alerts:
            content += "<div class='metrics'><h3>Active Alerts</h3><ul>"
            for alert in alerts:
                content += f"<li>{alert}</li>"
            content += "</ul></div>"
        
        return content
    
    def _generate_report_html_content(self, **kwargs) -> str:
        """Generate HTML content for report notifications."""
        report_type = kwargs.get('report_type', 'System Report')
        report_data = kwargs.get('report_data', {})
        attachments = kwargs.get('attachments', [])
        
        content = f"""
        <h2>📊 {report_type}</h2>
        <p>Your requested report has been generated and is attached to this email.</p>
        """
        
        if report_data:
            content += "<div class='metrics'><h3>Report Summary</h3><table>"
            for key, value in report_data.items():
                content += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"
            content += "</table></div>"
        
        if attachments:
            content += "<div class='metrics'><h3>Attachments</h3><ul>"
            for attachment in attachments:
                filename = os.path.basename(attachment) if isinstance(attachment, str) else attachment.get('filename', 'Unknown')
                content += f"<li>📎 {filename}</li>"
            content += "</ul></div>"
        
        return content
    
    def _generate_plain_text_template(self, template_type: str, **kwargs) -> str:
        """
        Generate plain text email template.
        
        Args:
            template_type: Type of template
            **kwargs: Template variables
            
        Returns:
            Plain text template string
        """
        header = f"""
SHIOL+ Powerball Prediction System
{'=' * 50}
Notification: {template_type.title()}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
{'=' * 50}

"""
        
        if template_type == 'pipeline':
            content = self._generate_pipeline_text_content(**kwargs)
        elif template_type == 'error':
            content = self._generate_error_text_content(**kwargs)
        elif template_type == 'health':
            content = self._generate_health_text_content(**kwargs)
        elif template_type == 'report':
            content = self._generate_report_text_content(**kwargs)
        else:
            content = f"Message: {kwargs.get('message', 'No message provided')}\n"
        
        footer = f"""
{'=' * 50}
This is an automated notification from SHIOL+ Powerball Prediction System.
For support, please contact your system administrator.
"""
        
        return header + content + footer
    
    def _generate_pipeline_text_content(self, **kwargs) -> str:
        """Generate plain text content for pipeline notifications."""
        status = kwargs.get('status', 'unknown')
        execution_time = kwargs.get('execution_time', 'unknown')
        results = kwargs.get('results', {})
        summary = kwargs.get('summary', {})
        
        content = f"""PIPELINE EXECUTION REPORT
Status: {status.upper()}
Execution Time: {execution_time}
Steps Completed: {summary.get('successful_steps', 0)}/{summary.get('total_steps', 0)}
Success Rate: {summary.get('success_rate', '0%')}
Pipeline Health: {summary.get('pipeline_health', 'unknown')}

"""
        
        if results:
            content += "STEP RESULTS:\n"
            for step_name, step_result in results.items():
                step_status = step_result.get('status', 'unknown')
                status_symbol = "[OK]" if step_status == 'success' else "[FAIL]"
                content += f"  {status_symbol} {step_name.replace('_', ' ').title()}: {step_status}\n"
            content += "\n"
        
        return content
    
    def _generate_error_text_content(self, **kwargs) -> str:
        """Generate plain text content for error notifications."""
        error_type = kwargs.get('error_type', 'Unknown Error')
        error_message = kwargs.get('error_message', 'No error message provided')
        context = kwargs.get('context', {})
        
        content = f"""ERROR ALERT
Error Type: {error_type}
Error Message: {error_message}

"""
        
        if context:
            content += "ERROR CONTEXT:\n"
            for key, value in context.items():
                content += f"  {key.replace('_', ' ').title()}: {value}\n"
            content += "\n"
        
        return content
    
    def _generate_health_text_content(self, **kwargs) -> str:
        """Generate plain text content for health notifications."""
        health_status = kwargs.get('health_status', 'unknown')
        metrics = kwargs.get('metrics', {})
        alerts = kwargs.get('alerts', [])
        
        content = f"""SYSTEM HEALTH ALERT
Health Status: {health_status.upper()}

"""
        
        if metrics:
            content += "SYSTEM METRICS:\n"
            for metric, value in metrics.items():
                content += f"  {metric.replace('_', ' ').title()}: {value}\n"
            content += "\n"
        
        if alerts:
            content += "ACTIVE ALERTS:\n"
            for alert in alerts:
                content += f"  - {alert}\n"
            content += "\n"
        
        return content
    
    def _generate_report_text_content(self, **kwargs) -> str:
        """Generate plain text content for report notifications."""
        report_type = kwargs.get('report_type', 'System Report')
        report_data = kwargs.get('report_data', {})
        attachments = kwargs.get('attachments', [])
        
        content = f"""REPORT NOTIFICATION
Report Type: {report_type}

Your requested report has been generated and is attached to this email.

"""
        
        if report_data:
            content += "REPORT SUMMARY:\n"
            for key, value in report_data.items():
                content += f"  {key.replace('_', ' ').title()}: {value}\n"
            content += "\n"
        
        if attachments:
            content += "ATTACHMENTS:\n"
            for attachment in attachments:
                filename = os.path.basename(attachment) if isinstance(attachment, str) else attachment.get('filename', 'Unknown')
                content += f"  - {filename}\n"
            content += "\n"
        
        return content
    
    def _send_email_with_retry(self, recipients: List[str], subject: str, 
                              html_body: str, text_body: str, 
                              attachments: Optional[List[Union[str, Dict]]] = None) -> Dict[str, Any]:
        """
        Send email with retry logic.
        
        Args:
            recipients: List of recipient email addresses
            subject: Email subject
            html_body: HTML email body
            text_body: Plain text email body
            attachments: Optional list of attachments
            
        Returns:
            Dict with send results
        """
        send_results = {
            'success': False,
            'recipients_sent': [],
            'recipients_failed': [],
            'attempts': 0,
            'error': None
        }
        
        for attempt in range(self.max_retry_attempts):
            send_results['attempts'] = attempt + 1
            
            try:
                logger.info(f"Sending email attempt {attempt + 1}/{self.max_retry_attempts}")
                
                # Create message
                msg = MIMEMultipart('alternative')
                msg['Subject'] = subject
                msg['From'] = self.smtp_username
                msg['To'] = ', '.join(recipients)
                
                # Add text and HTML parts
                text_part = MIMEText(text_body, 'plain', 'utf-8')
                html_part = MIMEText(html_body, 'html', 'utf-8')
                msg.attach(text_part)
                msg.attach(html_part)
                
                # Add attachments
                if attachments:
                    for attachment in attachments:
                        if isinstance(attachment, str):
                            # File path
                            if os.path.exists(attachment):
                                with open(attachment, 'rb') as f:
                                    part = MIMEBase('application', 'octet-stream')
                                    part.set_payload(f.read())
                                encoders.encode_base64(part)
                                part.add_header(
                                    'Content-Disposition',
                                    f'attachment; filename= {os.path.basename(attachment)}'
                                )
                                msg.attach(part)
                        elif isinstance(attachment, dict):
                            # Dict with filename and content
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(attachment.get('content', b''))
                            encoders.encode_base64(part)
                            part.add_header(
                                'Content-Disposition',
                                f'attachment; filename= {attachment.get("filename", "attachment")}'
                            )
                            msg.attach(part)
                
                # Send email
                if self.smtp_use_ssl:
                    context = ssl.create_default_context()
                    server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context)
                else:
                    server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                    if self.smtp_use_tls:
                        context = ssl.create_default_context()
                        server.starttls(context=context)
                
                if self.smtp_username and self.smtp_password:
                    server.login(self.smtp_username, self.smtp_password)
                
                # Send to each recipient
                failed_recipients = server.send_message(msg, to_addrs=recipients)
                server.quit()
                
                # Process results
                send_results['recipients_sent'] = [r for r in recipients if r not in failed_recipients]
                send_results['recipients_failed'] = list(failed_recipients.keys()) if failed_recipients else []
                
                if send_results['recipients_sent']:
                    send_results['success'] = True
                    # Record successful sends for rate limiting
                    for recipient in send_results['recipients_sent']:
                        self._record_email_sent(recipient)
                    
                    logger.info(f"Email sent successfully to {len(send_results['recipients_sent'])} recipients")
                    break  # Success, exit retry loop
                
            except smtplib.SMTPAuthenticationError as e:
                error_msg = f"SMTP authentication failed: {str(e)}"
                send_results['error'] = error_msg
                logger.error(error_msg)
                break  # Don't retry authentication errors
                
            except smtplib.SMTPRecipientsRefused as e:
                error_msg = f"Recipients refused: {str(e)}"
                send_results['error'] = error_msg
                send_results['recipients_failed'] = list(e.recipients.keys())
                logger.error(error_msg)
                break  # Don't retry recipient errors
                
            except (smtplib.SMTPConnectError, smtplib.SMTPServerDisconnected) as e:
                error_msg = f"SMTP connection error: {str(e)}"
                send_results['error'] = error_msg
                logger.error(f"{error_msg} (attempt {attempt + 1}/{self.max_retry_attempts})")
                
                if attempt < self.max_retry_attempts - 1:
                    delay = self.retry_delay_seconds * (self.retry_backoff_multiplier ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                
            except Exception as e:
                error_msg = f"Unexpected error sending email: {str(e)}"
                send_results['error'] = error_msg
                logger.error(f"{error_msg} (attempt {attempt + 1}/{self.max_retry_attempts})")
                
                if attempt < self.max_retry_attempts - 1:
                    delay = self.retry_delay_seconds * (self.retry_backoff_multiplier ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
        
        # Record notification attempt
        notification_record = {
            'timestamp': datetime.now().isoformat(),
            'subject': subject,
            'recipients': recipients,
            'success': send_results['success'],
            'attempts': send_results['attempts'],
            'error': send_results['error']
        }
        self.notification_history.append(notification_record)
        
        return send_results
    
    def send_pipeline_notification(self, pipeline_result: Dict[str, Any],
                                 recipients: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Send pipeline execution notification.
        
        Args:
            pipeline_result: Pipeline execution results
            recipients: Optional list of specific recipients
            
        Returns:
            Dict with notification results
        """
        try:
            status = pipeline_result.get('status', 'unknown')
            notification_level = 'success' if status == 'success' else 'error' if status == 'failed' else 'warning'
            
            # Filter recipients
            valid_recipients = self._filter_recipients(recipients, notification_level)
            if not valid_recipients:
                return {'success': False, 'message': 'No valid recipients or notification level disabled'}
            
            # Generate subject
            status_emoji = "✅" if status == 'success' else "❌" if status == 'failed' else "⚠️"
            subject = f"{status_emoji} SHIOL+ Pipeline Execution {status.title()}"
            
            # Generate email content
            html_body = self._generate_html_template('pipeline', **pipeline_result)
            text_body = self._generate_plain_text_template('pipeline', **pipeline_result)
            
            # Send email
            send_result = self._send_email_with_retry(valid_recipients, subject, html_body, text_body)
            
            logger.info(f"Pipeline notification sent: {send_result['success']}")
            return send_result
            
        except Exception as e:
            logger.error(f"Error sending pipeline notification: {e}")
            return {'success': False, 'error': str(e)}
    
    def send_error_alert(self, error_type: str, error_message: str,
                        context: Optional[Dict[str, Any]] = None,
                        traceback_info: Optional[str] = None,
                        recipients: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Send error alert notification.
        
        Args:
            error_type: Type of error
            error_message: Error message
            context: Optional error context
            traceback_info: Optional traceback information
            recipients: Optional list of specific recipients
            
        Returns:
            Dict with notification results
        """
        try:
            # Filter recipients
            valid_recipients = self._filter_recipients(recipients, 'error')
            if not valid_recipients:
                return {'success': False, 'message': 'No valid recipients or error notifications disabled'}
            
            # Generate subject
            subject = f"🚨 SHIOL+ Error Alert: {error_type}"
            
            # Prepare template data
            template_data = {
                'error_type': error_type,
                'error_message': error_message,
                'context': context or {},
                'traceback': traceback_info
            }
            
            # Generate email content
            html_body = self._generate_html_template('error', **template_data)
            text_body = self._generate_plain_text_template('error', **template_data)
            
            # Send email
            send_result = self._send_email_with_retry(valid_recipients, subject, html_body, text_body)
            
            logger.info(f"Error alert sent: {send_result['success']}")
            return send_result
            
        except Exception as e:
            logger.error(f"Error sending error alert: {e}")
            return {'success': False, 'error': str(e)}
    
    def send_health_alert(self, health_status: str, metrics: Dict[str, Any],
                         alerts: Optional[List[str]] = None,
                         recipients: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Send system health alert notification.
        
        Args:
            health_status: Overall health status (healthy, warning, critical)
            metrics: System metrics dictionary
            alerts: List of active alerts
            recipients: Optional list of specific recipients
            
        Returns:
            Dict with notification results
        """
        try:
            # Determine notification level
            notification_level = 'success' if health_status == 'healthy' else 'warning' if health_status == 'warning' else 'error'
            
            # Filter recipients
            valid_recipients = self._filter_recipients(recipients, notification_level)
            if not valid_recipients:
                return {'success': False, 'message': 'No valid recipients or notification level disabled'}
            
            # Generate subject
            status_emoji = "💚" if health_status == 'healthy' else "⚠️" if health_status == 'warning' else "🔴"
            subject = f"{status_emoji} SHIOL+ System Health: {health_status.title()}"
            
            # Prepare template data
            template_data = {
                'health_status': health_status,
                'metrics': metrics,
                'alerts': alerts or []
            }
            
            # Generate email content
            html_body = self._generate_html_template('health', **template_data)
            text_body = self._generate_plain_text_template('health', **template_data)
            
            # Send email
            send_result = self._send_email_with_retry(valid_recipients, subject, html_body, text_body)
            
            logger.info(f"Health alert sent: {send_result['success']}")
            return send_result
            
        except Exception as e:
            logger.error(f"Error sending health alert: {e}")
            return {'success': False, 'error': str(e)}
    
    def send_report(self, report_type: str, report_data: Dict[str, Any],
                   attachments: Optional[List[Union[str, Dict]]] = None,
                   recipients: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Send report with attachments.
        
        Args:
            report_type: Type of report
            report_data: Report data dictionary
            attachments: List of file paths or attachment dicts
            recipients: Optional list of specific recipients
            
        Returns:
            Dict with notification results
        """
        try:
            # Filter recipients (reports are usually 'success' level)
            valid_recipients = self._filter_recipients(recipients, 'success')
            if not valid_recipients:
                return {'success': False, 'message': 'No valid recipients or success notifications disabled'}
            
            # Generate subject
            subject = f"📊 SHIOL+ Report: {report_type}"
            
            # Prepare template data
            template_data = {
                'report_type': report_type,
                'report_data': report_data,
                'attachments': attachments or []
            }
            
            # Generate email content
            html_body = self._generate_html_template('report', **template_data)
            text_body = self._generate_plain_text_template('report', **template_data)
            
            # Send email with attachments
            send_result = self._send_email_with_retry(valid_recipients, subject, html_body, text_body, attachments)
            
            logger.info(f"Report sent: {send_result['success']}")
            return send_result
            
        except Exception as e:
            logger.error(f"Error sending report: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_notification_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent notification history.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of notification records
        """
        return list(self.notification_history)[-limit:]
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """
        Get current rate limit status for all recipients.
        
        Returns:
            Dict with rate limit status
        """
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        status = {
            'timestamp': now.isoformat(),
            'limits': {
                'max_per_hour': self.max_emails_per_hour,
                'max_per_day': self.max_emails_per_day
            },
            'recipients': {}
        }
        
        for recipient, history in self.rate_limit_tracker.items():
            # Clean old entries
            while history and history[0] < day_ago:
                history.popleft()
            
            emails_last_hour = sum(1 for timestamp in history if timestamp > hour_ago)
            emails_last_day = len(history)
            
            status['recipients'][recipient] = {
                'emails_last_hour': emails_last_hour,
                'emails_last_day': emails_last_day,
                'remaining_hour': max(0, self.max_emails_per_hour - emails_last_hour),
                'remaining_day': max(0, self.max_emails_per_day - emails_last_day),
                'blocked': emails_last_hour >= self.max_emails_per_hour or emails_last_day >= self.max_emails_per_day
            }
        
        return status


# Factory function for creating NotificationEngine instances
def create_notification_engine(config_path: str = "config/config.ini") -> NotificationEngine:
    """
    Factory function to create a NotificationEngine instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        NotificationEngine instance
    """
    try:
        engine = NotificationEngine(config_path=config_path)
        logger.info("NotificationEngine created successfully")
        return engine
        
    except Exception as e:
        logger.error(f"Failed to create NotificationEngine: {e}")
        raise


# Example usage and testing
if __name__ == "__main__":
    """Example usage of NotificationEngine."""
    
    # Create notification engine
    engine = create_notification_engine()
    
    # Validate configuration
    validation = engine.validate_config()
    print("Configuration Validation:")
    print(f"  Valid: {validation['valid']}")
    if validation['errors']:
        print(f"  Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"  Warnings: {validation['warnings']}")
    
    # Test SMTP connection
    if validation['valid']:
        connection_test = engine.test_connection()
        print(f"\nSMTP Connection Test:")
        print(f"  Success: {connection_test['success']}")
        if connection_test['error']:
            print(f"  Error: {connection_test['error']}")
    
    # Example pipeline notification
    pipeline_result = {
        'status': 'success',
        'execution_time': '0:05:23',
        'results': {
            'data_update': {'status': 'success'},
            'prediction_generation': {'status': 'success'},
            'validation': {'status': 'success'}
        },
        'summary': {
            'successful_steps': 3,
            'total_steps': 3,
            'success_rate': '100%',
            'pipeline_health': 'healthy'
        }
    }
    
    print(f"\nExample pipeline notification would be sent to: {engine.notification_recipients}")
    
    # Show rate limit status
    rate_status = engine.get_rate_limit_status()
    print(f"\nRate Limit Status:")
    print(f"  Max per hour: {rate_status['limits']['max_per_hour']}")
    print(f"  Max per day: {rate_status['limits']['max_per_day']}")
                