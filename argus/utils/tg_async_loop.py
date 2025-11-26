import asyncio
import threading

# Глобальный loop
_loop = None

def init_tg_async_loop():
    # Отдельный event loop для отправки сообщений в телеграм из колбека
    # Иначе если отправка зависает, то она вешает весь колбек
    global _loop
    _loop = asyncio.new_event_loop()
    threading.Thread(target=_loop.run_forever, daemon=True).start()

def run_async(func, *args, **kwargs):
    # Закидываем асинхронную таску в loop
    global _loop
    if _loop is None:
        raise RuntimeError("Async loop not initialized")

    return asyncio.run_coroutine_threadsafe(
        func(*args, **kwargs),
        _loop,
    )