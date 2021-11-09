# Object detecton using OpenVINO and YOLO v4

| Details                 |               |
|-------------------------|---------------|
| Programming Language:   |[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/) |
| Intel OpenVINO ToolKit: |[![OpenVINO 2020.3](https://img.shields.io/badge/openvino-2020.3-blue.svg)](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html)|
| Hardware Used:          | Raspberry Pi B+ |
| Device:                 | CPU or Intel Neural Cumpute Stick 2 or other Intel VPUs devices |

![Detected](https://github.com/rydikov/argus/raw/main/res/detected.jpg)

Argus application uses Deep Learning/Machine Learning to recognize objects on sources stream. 
Sources can be cameras and videos.
Application save frames with detection objecs and has the ability to telegam alert.

By utilizing pre-trained models and Intel OpenVINO toolkit with OpenCV. 

This application executes parallel threads for capture frames from sources and make async infer requests for objects detection.

**What is OpenVino?**

OpenVino (OpenVisual Inferencing and Neural Network Optimization) is toolkit to develop Deep Learning Application especially for Computer Vision by Intel. OpenVino Enables deep learning inference at the edge and supports heterogeneous execution across computer vision accelerators—CPU, GPU, Intel® Movidius™ Neural Compute Stick, and FPGA—using a common API. [read more](https://docs.openvinotoolkit.org/)

## Architecture

In different threads, frames put to the queue from cameras.
In the main thread, frames gets from queue and send to asynchronous recognition. Also in main thread results are received and processed.

## Hardware Requirement

- Minimum Intel Gen 6 processors or Raspberry with Neural Compute Stick

## Installtion for develop

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
source /opt/intel/openvino_2021.3.394/bin/setupvars.sh
python run.py development.yml
```

## Installation OpenVINO on Raspbery Pi

!!! Precompiled toolkit for Raspbian don't support ngraph and you must install OpenVINO manually. !!!

1. Set up build environment and install build tools
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install build-essential
```
2. Install CMake from source
```bash
cd ~/
wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4.tar.gz
tar xvzf cmake-3.14.4.tar.gz
cd ~/cmake-3.14.4
./bootstrap
make -j4 && sudo make install
```
3. Install OpenCV from source
```bash
sudo apt install git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python3-scipy libatlas-base-dev
cd ~/
git clone --depth 1 --branch 4.5.2 https://github.com/opencv/opencv.git
cd opencv && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local ..
make -j4 && sudo make install
```
4. Download source code and install dependencies
```bash
cd ~/
git clone --depth 1 --branch 2021.3 https://github.com/openvinotoolkit/openvino.git
cd ~/openvino
git submodule update --init --recursive
sh ./install_build_dependencies.sh
cd ~/openvino/inference-engine/ie_bridges/python/
pip3 install -r requirements.txt
```
5. Start CMake build
```bash
export OpenCV_DIR=/usr/local/lib/cmake/opencv4
cd ~/openvino
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=/home/pi/openvino_dist \
-DENABLE_MKL_DNN=OFF \
-DENABLE_CLDNN=OFF \
-DENABLE_GNA=OFF \
-DENABLE_SSE42=OFF \
-DTHREADING=SEQ \
-DENABLE_OPENCV=OFF \
-DNGRAPH_PYTHON_BUILD_ENABLE=ON \
-DNGRAPH_ONNX_IMPORT_ENABLE=ON \
-DENABLE_PYTHON=ON \
-DPYTHON_EXECUTABLE=$(which python3.7) \
-DPYTHON_LIBRARY=/usr/lib/arm-linux-gnueabihf/libpython3.7m.so \
-DPYTHON_INCLUDE_DIR=/usr/include/python3.7 \
-DWITH_GSTREAMER=ON \
-DCMAKE_CXX_FLAGS=-latomic ..
make -j4 && sudo make install
```
6. Configure the Intel® Neural Compute Stick 2 Linux USB Driver
```bash
sudo usermod -a -G users "$(whoami)"
source /home/pi/openvino_dist/bin/setupvars.sh
sh /home/pi/openvino_dist/install_dependencies/install_NCS_udev_rules.sh
```
7. Verify nGraph module binding to Python
```bash
cd /home/pi/openvino_dist/deployment_tools/inference_engine/samples/python/object_detection_sample_ssd
python3 object_detection_sample_ssd.py -h
```

8. Clone project
```bash
git clone git@github.com:rydikov/argus.git
cd argus
git lfs pull
```

9. Install dependencies
```bash
pip3 install -r requirements.txt
```

10. Add symlinks to supervisor and nginx config. I, also, recommend creating a private repository for production configs.

My private repository contain files:
* nginx.conf - Nginx to view images
* supervisord.conf – On raspberry app started with supervisor
* loki.yml - I'm use Cloud Grafana (free) for visualize metrics and alerting. 
* production.yml - Production config

Grafana looks like:

![Detected](https://github.com/rydikov/argus/raw/main/res/grafana.png)

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
|     save_every_n_frame |          | Save every N frame                                                       |
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
    save_every_n_frame: 15
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
    save_every_n_frame: 0
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
- [OpenVino](https://docs.openvinotoolkit.org/latest/index.html)
- https://stackoverflow.com/questions/66831806/loading-openvino-python-library-on-raspebrry-pi-4

