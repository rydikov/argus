from datetime import datetime

from pythonjsonlogger.jsonlogger import JsonFormatter


class RFC3339JsonFormatter(JsonFormatter):
    """Add timestamp in RFC3339 for Grafana"""
    def add_fields(self, log_record, record, message_dict):
        super(RFC3339JsonFormatter, self).add_fields(log_record, record, message_dict)
        if not log_record.get('timestamp'):
            now = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            log_record['timestamp'] = now
