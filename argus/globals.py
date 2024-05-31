import os
import yaml

from argus.services.alarm_system import AlarmSystemService
from argus.services.telegram import TelegramService
from argus.services.recognizer import OpenVinoRecognizer

dir_path = os.path.dirname(os.path.realpath(__file__))

config_path = os.getenv('CONFIG_PATH', os.path.join(dir_path, '../development.yml'))

with open(os.path.join(dir_path, config_path)) as f:
    config = yaml.safe_load(f)

if config.get('telegram') is not None:
    telegram_service = TelegramService(
        config['telegram']['bot_token'], 
        config['telegram']['bot_chat_id']
    )
else:
    telegram_service = None

alarm_system_service = AlarmSystemService()

recognizer = OpenVinoRecognizer(config['recognizer'], telegram_service)