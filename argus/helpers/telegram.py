import requests

from datetime import datetime, timedelta


class Telegram:
    def __init__(self, config):
        self.bot_token = config['telegram_bot_token']
        self.bot_chat_id = config['telegram_bot_chat_id']

        self.time_of_the_last_attempt = None
        self.silent_until_time = datetime.now()

    def send_message(self, message):
        send_text = 'https://api.telegram.org/bot{}/sendMessage?chat_id={}&parse_mode=Markdown&text={}'.format(
            self.bot_token,
            self.bot_chat_id,
            message
        )
        response = requests.get(send_text)
        return response.json()

    def send_and_be_silent(self, message, silent_time=timedelta(minutes=30)):
        self.time_of_the_last_attempt = datetime.now()
        if self.time_of_the_last_attempt > self.silent_until_time:
            self.silent_until_time = datetime.now() + silent_time
            return self.send_message(message)
