import os
import yaml

from argus.services.telegram import TelegramService
from argus.services.recognizer import OpenVinoRecognizer
from argus.services.mqtt import MQTTService

dir_path = os.path.dirname(os.path.realpath(__file__))

config_path = os.getenv('CONFIG_PATH', os.path.join(dir_path, '../development.yml'))

with open(os.path.join(dir_path, config_path)) as f:
    config = yaml.safe_load(f)

state_dir = config['app']['state_dir']
if not os.path.exists(state_dir):
    os.makedirs(state_dir, mode=0o777)

if config.get('telegram_bot') is not None:
    telegram_service = TelegramService(
        config['telegram_bot']['token'], 
        config['telegram_bot']['chat_id']
    )
else:
    telegram_service = None

if config.get('mqtt') is not None:
    mqtt_service = MQTTService(
        config['mqtt']['hostname'], 
        config['mqtt']['port']
    )
else:
    mqtt_service = None


recognizer = OpenVinoRecognizer(
    config['recognizer'], 
    telegram_service,
    mqtt_service
)