# import cv2
# import requests


# class TelegramService:
#     def __init__(self, bot_token, bot_chat_id):
#         self.bot_token = bot_token
#         self.bot_chat_id = bot_chat_id

#     def send_message(self, message):
#         send_text = 'https://api.telegram.org/bot{}/sendMessage?chat_id={}&parse_mode=Markdown&text={}'.format(
#             self.bot_token,
#             self.bot_chat_id,
#             message
#         )
#         response = requests.get(send_text)
#         return response.json()

#     def send_frame(self, frame, message):
#         _, img = cv2.imencode('.JPEG', frame)
#         data = {"chat_id": self.bot_chat_id, "caption": message}
#         url = f'https://api.telegram.org/bot{self.bot_token}/sendPhoto'
#         response = requests.post(url, data=data, files={'photo': img.tobytes()})
#         return response.json()



import cv2
from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.types import FSInputFile
from tempfile import NamedTemporaryFile


class TelegramService:
    def __init__(self, bot_token, bot_chat_id):
        self.bot_token = bot_token
        self.bot_chat_id = bot_chat_id
        self.bot = Bot(
            token=self.bot_token,
            default=DefaultBotProperties(parse_mode="Markdown")
        )

    async def send_message(self, message):
        return await self.bot.send_message(chat_id=self.bot_chat_id, text=message)

    async def send_frame(self, frame, message):
        _, img = cv2.imencode('.jpg', frame)
        with NamedTemporaryFile(suffix=".jpg") as temp:
            temp.write(img.tobytes())
            temp.flush()
            photo = FSInputFile(temp.name)
            return await self.bot.send_photo(chat_id=self.bot_chat_id, photo=photo, caption=message)
