#!/usr/bin/env python3

import argparse
import socket
import sys


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8888
DEFAULT_TIMEOUT = 5.0

COMMANDS = {
    "restart": "Перезапустить систему видеообнаружения",
    "reset": "Разрешить снова отправлять уведомления",
    "get_photos": "Запросить отправку фото/кадров",
}


def send_bus_message(host: str, port: int, message: str, timeout: float) -> str:
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(message.encode("utf-8"))
            received = sock.recv(1024).decode("utf-8").strip()
    except ConnectionRefusedError:
        raise RuntimeError("connection refused")
    except socket.timeout:
        raise RuntimeError("timeout")
    except OSError as exc:
        raise RuntimeError(f"socket error: {exc}") from exc

    if received != "OK":
        raise RuntimeError(f"server error: {received!r}")

    return received


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="video-signal",
        description="CLI для отправки команд",
    )

    parser.add_argument(
        "command",
        choices=COMMANDS.keys(),
        help="Команда для argus",
    )

    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Host сервера, по умолчанию {DEFAULT_HOST}",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port сервера, по умолчанию {DEFAULT_PORT}",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Таймаут соединения в секундах, по умолчанию {DEFAULT_TIMEOUT}",
    )

    args = parser.parse_args()

    try:
        response = send_bus_message(
            host=args.host,
            port=args.port,
            message=args.command,
            timeout=args.timeout,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())