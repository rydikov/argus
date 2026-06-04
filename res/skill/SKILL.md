# Video Detection Control Skill

## Purpose

Control the local video detection system through its TCP external signal interface.

The system accepts three commands:

- `restart` — restart the video detection system
- `reset` — clear notification throttlers, including the Telegram silence timer that suppresses new frames after a detection
- `get_photos` — request sending photos/frames from active frame queues

## CLI

Use the `video-signal` command.

Default connection:

- host: `127.0.0.1`
- port: `8888`
- timeout: `5` seconds

Command format:

```bash
argus_cli.py <command> --host <host> --port <port>
```

## Commands

### Restart video detection

```bash
argus_cli.py restart --host 127.0.0.1 --port 8888
```

Use when the user asks to restart the video detection system.

Russian examples:

- перезапусти argus
- перезапусти сервис обнаружения

### Reset notification throttling

```bash
argus_cli.py reset --host 127.0.0.1 --port 8888
```

Use when the user asks to reset notification limits, allow notifications again, or discard an unimportant detection and wait for the next one.

`reset` resets the "silence" period during which new frames are not sent to Telegram after a detection. This is useful when the latest detection turned out to be unimportant: run `reset`, then wait for the next detection to send fresh frames.

Russian examples:

- разреши снова отправлять уведомления
- это неважный детект, сбрось и жди следующий

### Get photos / frames

```bash
argus_cli.py get_photos --host 127.0.0.1 --port 8888
```

Use when the user asks to request current photos or frames from the video detection system.

Russian examples:

- получи фото с камер
- запроси кадры
- отправь текущие фото
- покажи последние снимки

## Success condition

The command is successful only when stdout is:

```text
OK
```

## Failure handling

If the command exits with a non-zero code, report the error from stderr.

Common errors:

- `connection refused` — video detection server is not running or wrong host/port
- `timeout` — server is unavailable or not responding
- `server error` — server returned a response other than `OK`

## Safety

Do not call `restart` unless the user explicitly asks to restart the system.

Prefer `reset` for notification throttling problems and for unimportant detections where the user wants to wait for the next fresh Telegram frames.

Prefer `get_photos` for diagnostic photo/frame requests.
