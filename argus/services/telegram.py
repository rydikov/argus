import cv2
import requests


class TelegramService:
    def __init__(self, bot_token, bot_chat_id):
        self.bot_token = bot_token
        self.bot_chat_id = bot_chat_id

    def send_message(self, message):
        send_text = 'https://api.telegram.org/bot{}/sendMessage?chat_id={}&parse_mode=Markdown&text={}'.format(
            self.bot_token,
            self.bot_chat_id,
            message
        )
        response = requests.get(send_text)
        return response.json()

    def send_frame(self, frame, message):
        _, img = cv2.imencode('.JPEG', frame)
        data = {"chat_id": self.bot_chat_id, "caption": message}
        url = f'https://api.telegram.org/bot{self.bot_token}/sendPhoto'
        response = requests.post(url, data=data, files={'photo': img.tobytes()})
        return response.json()

