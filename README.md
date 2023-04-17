# Argus

| Details                 |               |
|-------------------------|---------------|
| Neural network:         |[![YOLOv7](https://img.shields.io/badge/yolo-7m-blue)](https://github.com/WongKinYiu/yolov7) |
| Intel OpenVINO ToolKit: |[![OpenVINO 2021.4.2](https://img.shields.io/badge/openvino-2021.4-blue.svg)](https://docs.openvino.ai/2021.4/openvino_docs_install_guides_installing_openvino_raspbian.html)|
| Hardware Used:          | Raspberry Pi B+ |
| Device:                 | CPU or Intel Neural Cumpute Stick 2 or other Intel VPUs devices |

*Intel® Movidius ™ VPU based products, including Intel® Neural Compute Stick 2 (NCS2) are not supported in the latest release, OpenVINO 2022.3.1, but it will be added back in a future OpenVINO 2022.3.1 LTS update.*

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

0. Install [Intel OpenVINO ToolKit](https://software.seek.intel.com/openvino-toolkit)

1. Clone project and download models
```bash
git clone git@github.com:rydikov/argus.git
cd argus
git lfs pull
```

2. Install dependencies
```bash
cd argus
pip3 install -r requirements.txt
```

Run application with example config
```bash
export PYTHONPATH=$PYTHONPATH:/PROJECT_PWD/argus
source /opt/intel/openvino_2021/bin/setupvars.sh
python run.py development.yml
```

## Installation OpenVINO on Raspbery Pi

!!! Precompiled toolkit for Raspbian don't support ngraph and you must install OpenVINO manually. !!!

1. Set up build environment and install build tools
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install build-essential
```
2. Install OpenVINO
https://docs.openvino.ai/2021.4/openvino_docs_install_guides_installing_openvino_raspbian.html

3. Clone project
```bash
git clone git@github.com:rydikov/argus.git
cd argus
git lfs pull
```

4. Install dependencies
```bash
pip3 install -r requirements.txt
```

5. Add symlinks to supervisor and nginx config. I, also, recommend creating a private repository for production configs.

My private repository contain files:
* nginx.conf - Nginx to view images
* supervisord.conf – On raspberry app started with supervisor
* loki.yml - I'm use Cloud Grafana (free) for visualize metrics and alerting. 
* production.yml - Production config

Grafana looks like:

![Grafana](https://github.com/rydikov/argus/raw/main/res/grafana.png)

Nginx exapmle
```
server {
    listen   8080 default;
	server_name  localhost;

	access_log  /var/log/nginx/localhost.access.log;

	location / {
		root   /home/pi/timelapse/;
		autoindex  on;
		autoindex_localtime on;
                autoindex_exact_size off;
  }
}
```

Supervisor exapmle
```
[program:argus]
command=/bin/bash -c 'source /home/pi/openvino_dist/bin/setupvars.sh && sleep 5 && /usr/bin/python3.7 /home/pi/argus/argus/run.py /home/pi/argus-production-config/production.yml'
stdout_logfile=/home/pi/timelapse/argus.log
stdout_logfile_maxbytes=1MB
stdout_logfile_backups=10
stderr_logfile=/home/pi/timelapse/argus.err
stderr_logfile_maxbytes=1MB 
stderr_logfile_backups=10
redirect_stderr=true
autostart=true
autorestart=true
user=pi
environment=PYTHONPATH="$PYTHONPATH:/home/pi/argus"
```

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
          __path__: /home/pi/timelapse/argus.log
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

#### Sources secton

| Option                 | Required | Description                                                              |
|------------------------|----------|--------------------------------------------------------------------------|
| sources                | +        | Set of data sources                                                      |
|   source-name          | +        | Source name                                                              |
|     source             | +        | Source                                                                   |
|     stills_dir         | +        | Direcrory for saved frames                                               |
|     host_stills_uri    | +        | Web link to folder with frames                                           |
|     important_objects  | +        | Important objects. Mark an Alert if this objects detected on frame       |
|     other_objects      |          | Other objects. Mark if this objects detected on frame                    |
|     max_object_area    |          | Max object area for detecton                                             |
|     save_every_sec     |          | Save frame every N sec                                                   |
|     bfc                |          | Bad frame checker                                                        |
|       threshold        |          | Threshold for detecton. Set experimentally                               |
|       coords           |          | Coords for pattern image                                                 |
|       reverse_pixel    |          | Analyzed pixel. If Pixel is black - revert image                         |
|       template_path    |          | Template with pattern for analyze                                        |


Example for sources secton with all options:
```yaml
sources:
  first-cam:
    source: ../../demohd.mp4
    save_every_sec: 15
    stills_dir: ../../Stills/first
    host_stills_uri: http://localhost/Stills/first
    bfc:
      threshold: 17900000
      coords: [64, 104, 324, 352]
      reverse_pixel: [39, 0]
      template_path: ../../argus-production-config/res/2.jpg
    max_object_area: 15000
    important_objects:
      - person
  second-cam:
    source: ../../demo.mov
    save_every_sec: 0
    stills_dir: ../../Stills/second
    host_stills_uri: http://localhost/Stills/second
    max_object_area: 15000
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
|   device_name          | +        | Device for network                                                       |
|   num_requests         | +        | Num of requests for recognize. Usually 4 per one MYRIAD device           |

Example for recognizer secton with all options:
```yaml
recognizer:
  device_name: MYRIAD
  num_requests: 4
```

#### Telegram secton

| Option                 | Required | Description                                                              |
|------------------------|----------|--------------------------------------------------------------------------|
| telegram_bot           |          | Telegram section. Use it for alarming                                    |
|   token                |          | Token                                                                    |
|   chat_id              |          | ChatId                                                                   |

Example for telegram_bot secton with all options:

```yaml
telegram_bot:
  token: __token__
  chat_id: __chat_id__
```


## Credit

- AlexeyAB/darknet: [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
- WongKinYiu: [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- [OpenVino](https://docs.openvinotoolkit.org/latest/index.html)
- https://stackoverflow.com/questions/66831806/loading-openvino-python-library-on-raspebrry-pi-4
- https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build_raspbian.md
- https://github.com/bethusaisampath/YOLOs_OpenVINO/tree/main/YOLOv7
