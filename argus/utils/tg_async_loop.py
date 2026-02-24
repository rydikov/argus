import asyncio
import logging
import threading

# Глобальный loop
_loop = None
logger = logging.getLogger('json')

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

    future = asyncio.run_coroutine_threadsafe(
        func(*args, **kwargs),
        _loop,
    )

    task_name = getattr(func, '__qualname__', str(func))

    def _log_async_error(done_future):
        try:
            done_future.result()
        except Exception:
            logger.exception('Async task failed: %s', task_name)

    future.add_done_callback(_log_async_error)
    return future
