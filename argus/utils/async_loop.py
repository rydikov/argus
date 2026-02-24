import asyncio
import logging
import threading

# Отдельный event loop, который работает в фоне в daemon-thread.
# Нужен для запуска async-функций (Telegram/MQTT) из синхронных мест
# вроде callback-ов OpenVINO, где нельзя делать await напрямую.
_loop = None
logger = logging.getLogger('json')

def init_async_loop():
    # Создаем loop один раз и запускаем его в отдельном daemon-потоке.
    # run_forever держит loop активным, пока жив процесс.
    # Такой подход не блокирует основной pipeline распознавания кадров.
    global _loop
    _loop = asyncio.new_event_loop()
    threading.Thread(target=_loop.run_forever, daemon=True).start()

def run_async(func, *args, **kwargs):
    # Вызывается из синхронного кода:
    # 1) собираем coroutine через func(*args, **kwargs),
    # 2) передаем ее в фоновый loop потокобезопасно,
    # 3) получаем concurrent.futures.Future для контроля результата.
    #
    # Важно: без чтения результата future исключения внутри coroutine
    # часто "теряются" и не попадают в обычные логи.
    global _loop
    if _loop is None:
        raise RuntimeError("Async loop not initialized")

    future = asyncio.run_coroutine_threadsafe(
        func(*args, **kwargs),
        _loop,
    )

    task_name = getattr(func, '__qualname__', str(func))

    def _log_async_error(done_future):
        # done callback вызывается в момент завершения coroutine.
        # done_future.result() пробрасывает исключение, если задача упала,
        # и мы логируем полный stack trace через logger.exception.
        try:
            done_future.result()
        except Exception:
            logger.exception('Async task failed: %s', task_name)

    future.add_done_callback(_log_async_error)
    return future
