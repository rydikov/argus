# Argus

| Details                 |               |
|-------------------------|---------------|
| Neural network:         |[![YOLOv11](https://img.shields.io/badge/yolo-11-blue)](https://github.com/ultralytics/ultralytics) |
| CV Framework:           |[![OpenVINO 2025.4.1](https://img.shields.io/badge/openvino-2025.4-blue.svg)](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)|
| Hardware:               | Минимум Intel CPU 6-го поколения, PC с Intel GPU. |


![Detected](https://github.com/rydikov/argus/raw/main/res/detected.jpg)

Argus использует технологии компьютерного зрения для определение объектов.
В качестве источников могут использоваться RTSP потоки с камер.
Приложение сохраняет кадры с обнаруженными объектами и отправляет уведомления в Telegram.

## Архитектура

Кадры с камер добавляются в двухсторонние очереди в отдельных потоках.
В главном потоке кадры извлекаются из очередей и передаются на асинхронное распознавание. Всегда берется самый свежий кадр.
Также проверяется – есть ли распознанный кадр. Если есть, то в зависимости от обраруженных на нем объектов он сохраняется или выбрасывается.
Если были обнаружены important_objects, то происходит оповещение в Telegram. Оповещение происходит не чаще одного раза в 30 минут с каждого потока.
Если в конфигурации указаны настройки MQTT, то происходит публикация события.

Дополнительно можно настроить сохранение кадров на диск каждые N секунд.
Список распознаваемых объектов можно посмотреть в [models](models/coco.names)


## Запуск через Docker

Предполагается, что Docker уже устновлен на хостовой машине.

1. Склонируйте проект и перейдите в него
```bash
git clone git@github.com:rydikov/argus.git
cd argus
```

2. Запустите сборку
```bash
docker-compose build
```

3. Запустите проект
```bash
docker-compose up
```

По умолчанию проект запускается на модели yolo9s, в качестве потока испльзуется демонстрационное видео в каталоге res.

Для использования собственного файла конфигурации нужно создать файл c названием env и добавить в него переменную: CONFIG_PATH с полным путем к файлу.
Файл лучше разместить в дирректории data, т.к. она монтируется при запуске.
Пример:
```
CONFIG_PATH = /app/data/development.override.yml
```

Также можно установить приложение сразу на [хостовую машину](res/tips.md). 

### Настройка

### App section
| Option                 | Required | Description                                                                                |
|------------------------|----------|--------------------------------------------------------------------------------------------|
| app                    | +        | Основная секция                                                                            |


#### Sources secton

| Option                 | Required | Description                                                                                                                       |
|------------------------|----------|-----------------------------------------------------------------------------------------------------------------------------------|
| sources                | +        | Источники кадров.                                                                                                                 |
|   source-name          | +        | Название                                                                                                                          |
|     source             | +        | Ссылка на поток, например, rtsp адрес камеры                                                                                      |
|     stills_dir         | +        | Дирректория, куда будут сохраняться кадры                                                                                         |
|     host_stills_uri    |          | Ссылка web. Если nginx раздает сохраненные изображения, то в телграмм отправляется она, если не указана, то отправляется сам кадр |
|     important_objects  |          | Объекты, при обнаружении который будет оповещение в telegram. По умолчани: person.                                                |
|     other_objects      |          | Объекты, которые будут распозноваться и помечаться на изображении                                                                 |
|     save_every_sec     |          | Сохранять изображения каждые N секунд, если 0, то будут сохраняться только изображения с important_objects                        |


Пример с двумя камерами и всеми опциями для секций:

```yaml
sources:
  first-cam:
    source: rtsp://login:password@192.168.1.55:554/Streaming/Channels/101
    save_every_sec: 15
    stills_dir: /../../Stills/first
    important_objects:
      - person
  second-cam:
    source: rtsp://login:password@192.168.1.66:554/Streaming/Channels/101
    save_every_sec: 0
    stills_dir: ../../Stills/second
    host_stills_uri: http://localhost/Stills/second
    important_objects:
      - person
      - car
      - cow
    other_objects:
      - bicycle
      - motorcycle
      - bird
      - cat
      - dog
      - horse
```

#### Recognizer secton

| Option                 | Required | Description                                                                               |
|------------------------|----------|-------------------------------------------------------------------------------------------|
| recognizer             | +        | Секция распознования                                                                      |
|   model                |          | Модель (по умолчнанию yolov9s, более точные 9m и 9c можно сказать на вкладыке Releases)   |
|   device_name          | +        | Устройсто распознования (CPU, GPU, MYRIAD)                                                |
|   num_requests         | +        | Количество потоков распонования. Оптимально 4 для каждого MYRIAD. Подбирается эмпирически |

Example for recognizer secton with all options:
```yaml
recognizer:
  model: yolo11n
  device_name: MYRIAD
  num_requests: 4
```

#### Telegram secton (опциональная)

| Option                 | Required | Description                                                              |
|------------------------|----------|--------------------------------------------------------------------------|
| telegram_bot           |          | Telegram секция для опещения                                             |
|   token                |          | Токен                                                                    |
|   chat_id              |          | ID чата в который будет приходить оповещение                             |

#### Mqtt secton (опциональная)

| Option                 | Required | Description                                                              |
|------------------------|----------|--------------------------------------------------------------------------|
| mqtt                   |          | Telegram секция для опещения                                             |
|   hostname             |          | Хост mqtt                                                                |
|   port                 |          | Порт mqtt                                                                |


## Credit

- [OpenVino](https://docs.openvinotoolkit.org/latest/index.html)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
