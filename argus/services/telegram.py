import cv2
from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.types import FSInputFile
from tempfile import NamedTemporaryFile


class TelegramService:
    # Async Service for nonblocking recognized thread, run with run_async in helpers from sync callback
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
