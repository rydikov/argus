import threading
import asyncio

def run_async(func, *args, **kwargs):
    def runner():
        asyncio.run(func(*args, **kwargs))
    thread = threading.Thread(target=runner)
    thread.start()
    return thread