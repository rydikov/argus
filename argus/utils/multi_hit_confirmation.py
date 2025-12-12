import time
import logging

from collections import deque

logger = logging.getLogger('json')


class MultiHitConfirmation:
    def __init__(self, window_seconds=3, threshold=5):
        self.window = window_seconds
        self.threshold = threshold
        self.events = deque()

    def on_detect(self) -> bool:
        """
        Вызывается при каждом обнаружении.
        Возвращает True, если за последние N секунд набралось threshold событий.
        После этого сбрасывает счётчик, чтобы избежать спама.
        """
        logger.info(f'Events in deque: {len(self.events)}')

        now = time.time()

        # Добавляем новое событие
        self.events.append(now)

        # Удаляем старые события
        while self.events and now - self.events[0] > self.window:
            self.events.popleft()

        # Если накопилось нужное количество — подтверждаем
        if len(self.events) >= self.threshold:
            return True

        return False
