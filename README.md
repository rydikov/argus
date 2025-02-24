# Argus

| Details                 |               |
|-------------------------|---------------|
| Neural network:         |[![YOLOv9](https://img.shields.io/badge/yolo-9-blue)](https://github.com/WongKinYiu/yolov9) |
| Intel OpenVINO ToolKit: |[![OpenVINO 2022.3.2](https://img.shields.io/badge/openvino-2022.3-blue.svg)](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)|
| Hardware Used:          | Mini PC       |
| Device:                 | CPU or Intel Neural Cumpute Stick 2 or other Intel VPUs devices |


![Detected](https://github.com/rydikov/argus/raw/main/res/detected.jpg)

Argus application uses Deep Learning/Machine Learning to recognize objects on sources stream. 
The Sources can be cameras and videos.
The Application saves frames with detection objecs and has the ability to telegam alerts.

By utilizing pre-trained models and Intel OpenVINO toolkit with OpenCV. 

This application executes parallel threads to capture frames from different sources and make async infer requests for object detections.

**What is OpenVino?**

OpenVino (OpenVisual Inferencing and Neural Network Optimization) is toolkit to develop Deep Learning Application especially for Computer Vision by Intel. OpenVino Enables deep learning inference at the edge and supports heterogeneous execution across computer vision accelerators—CPU, GPU, Intel® Movidius™ Neural Compute Stick, and FPGA—using a common API. [read more](https://docs.openvinotoolkit.org/)

## Architecture

The frames put to the queue from cameras in different threads.
In the main thread, the frames get from a queue and send to asynchronous recognition. Also in main thread the results are received and processed.

## Hardware Requirement

- Minimum Intel Gen 6 processors or Raspberry Pi with Neural Compute Stick

![Hardware](https://github.com/rydikov/argus/raw/main/res/hardware.jpg)

## Installation for develop

1. Clone project and download models
```bash
git clone git@github.com:rydikov/argus.git
cd argus
```

2. Copy models to project (optional)

3. Build docker image
```bash
docker-compose build
```

Run application
```bash
docker-compose up
```

## Installation OpenVINO without Docker


1. Set up build environment and install build tools
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install build-essential
```
2. Install OpenVINO
https://docs.openvino.ai/2022.3/openvino_docs_install_guides_install_dev_tools.html

3. Clone project and save models from release to models folder
```bash
git clone git@github.com:rydikov/argus.git
cd argus
```

4. Install dependencies
```bash
pip3 install -r requirements.txt
```

5. Add symlinks to supervisor and nginx config. I, also, recommend creating a private repository for production configs.

My private repository contain files:
* production.yml - Production config
* supervisord.conf – On raspberry app started with supervisor
* nginx.conf - Nginx to view images (optional)
* loki.yml - I'm use Cloud Grafana (free) for visualize metrics and alerting (optional)


Supervisor exapmle
```
[program:argus]
command=/bin/bash -c 'source /opt/intel/openvino_2022/setupvars.sh && sleep 5 && /home/xcy/argus/.env/bin/python /home/xcy/argus/argus/run.py'
stdout_logfile=/home/xcy/timelapse/argus.log
stdout_logfile_maxbytes=1MB
stdout_logfile_backups=10
stderr_logfile=/home/xcy/timelapse/argus.err
stderr_logfile_maxbytes=1MB 
stderr_logfile_backups=10
redirect_stderr=true
autostart=true
autorestart=true
user=pi
environment=PYTHONPATH="$PYTHONPATH:/home/pi/argus",CONFIG_PATH="/home/xcy/argus-production-config/production.yml"
```


Nginx exapmle
```
server {
    listen   8080 default;
	server_name  localhost;

	access_log  /var/log/nginx/localhost.access.log;

	location / {
		root   /home/xcy/timelapse/;
		autoindex  on;
		autoindex_localtime on;
    autoindex_exact_size off;
  }
}
```

Grafana looks like:

![Grafana](https://github.com/rydikov/argus/raw/main/res/grafana.png)

Loki exapmle
```
loki:
  configs:
  - name: integrations
    positions:
      filename: /tmp/positions.yaml
    scrape_configs:
    - job_name: argus
      static_configs:
      - targets: [localhost]
        labels:
          job: argus
          __path__: /home/xcy/timelapse/argus.log
      pipeline_stages:
      - json:
          expressions:
            func: func
            loglevel: levelname
            threadName: threadName
            timestamp: timestamp
      - timestamp:
          format: RFC3339
          source: timestamp
      - labels:
          loglevel:
          func:
          threadName:
    clients:
    - url: __url__
      basic_auth:
        username: __username__
        password: __password__
```

11. Reload supervisor


### App Config options

### App section
| Option                 | Required | Description                                                                                |
|------------------------|----------|--------------------------------------------------------------------------------------------|
| app                    | +        | Main section                                                                               |
| state_dir              | +        | Path to app state for store tockens etc                                                    |



#### Sources secton

| Option                 | Required | Description                                                                                |
|------------------------|----------|--------------------------------------------------------------------------------------------|
| sources                | +        | Set of data sources                                                                        |
|   source-name          | +        | Source name                                                                                |
|     source             | +        | Source                                                                                     |
|     stills_dir         | +        | Direcrory for saved frames                                                                 |
|     host_stills_uri    |          | Web link to folder with frames                                                             |
|     important_objects  |          | Important objects. Mark an Alert if this objects detected on frame. Default: person        |
|     other_objects      |          | Other objects. Mark if this objects detected on frame                                      |
|     save_every_sec     |          | Save frame every N sec                                                                     |


Example for sources secton with all options:
```yaml
sources:
  first-cam:
    source: ../../demohd.mp4
    save_every_sec: 15
    stills_dir: ../../Stills/first
    host_stills_uri: http://localhost/Stills/first
    important_objects:
      - person
  second-cam:
    source: ../../demo.mov
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

| Option                 | Required | Description                                                              |
|------------------------|----------|--------------------------------------------------------------------------|
| recognizer             | +        | Recognize section                                                        |
|   model                |          | Model (yolov9c (default))                     |
|   device_name          | +        | Device for network                                                       |
|   num_requests         | +        | Num of requests for recognize. Usually 4 per one MYRIAD device           |

Example for recognizer secton with all options:
```yaml
recognizer:
  model: yolov9c
  device_name: MYRIAD
  num_requests: 4
```

#### Telegram secton (optional)

| Option                 | Required | Description                                                              |
|------------------------|----------|--------------------------------------------------------------------------|
| telegram_bot           |          | Telegram section. Use it for alarming                                    |
|   token                |          | Token                                                                    |
|   chat_id              |          | ChatId                                                                   |

#### Aqara secton (optional)

| Option                 | Required | Description                                                              |
|------------------------|----------|--------------------------------------------------------------------------|
| aqara                  |          | Aqara section. Use it for run scenes                                     |
|   scene_id             |          |                                                                          |
|   app_id               |          |                                                                          |
|   app_key              |          |                                                                          |
|   key_id               |          |                                                                          |
|   account              |          |                                                                          |

Example for aqara secton with all options:

```yaml
aqara:
  scene_id: AL.52785555917472
  app_id: __app_id__
  app_key: __app_key__
  key_id: __key_id__
  account: example@example.com
```
Detail: https://opendoc.aqara.cn/en/docs/developmanual/authManagement/aqaraauthMode.html


## Credit

- [OpenVino](https://docs.openvinotoolkit.org/latest/index.html)
- [Yolov9](https://github.com/WongKinYiu/yolov9)
- [Aqara](https://developer.aqara.com/?lang=en)
