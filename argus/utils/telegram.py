import requests


class Telegram:
    def __init__(self, config):
        self.bot_token = config['token']
        self.bot_chat_id = config['chat_id']

    def send_message(self, message):
        send_text = 'https://api.telegram.org/bot{}/sendMessage?chat_id={}&parse_mode=Markdown&text={}'.format(
            self.bot_token,
            self.bot_chat_id,
            message
        )
        response = requests.get(send_text)
        return response.json()
