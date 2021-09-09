# Object detecton using OpenVINO and YOLO v4

| Details            |              |
|-----------------------|---------------|
| Programming Language: |  [![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/) |
| Intel OpenVINO ToolKit: |[![OpenVINO 2020.3](https://img.shields.io/badge/openvino-2020.3-blue.svg)](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html)|
| Hardware Used: | Raspberry Pi B+ |
| Device: | CPU or Intel Neural Cumpute Stick 2 or other Intel VPUs devices |



Argus application uses Deep Learning/Machine Learning to recognize objects on sources stream. 
Sources can be cameras and videos.
Application save frames with detection objecs and has the ability to telegam alert.

By utilizing pre-trained models and Intel OpenVINO toolkit with OpenCV. 

This application executes parallel threads for capture frames from sources and make async infer requests for objects detection.

**What is OpenVino?**

OpenVino (OpenVisual Inferencing and Neural Network Optimization) is toolkit to develop Deep Learning Application especially for Computer Vision by Intel. OpenVino Enables deep learning inference at the edge and supports heterogeneous execution across computer vision accelerators—CPU, GPU, Intel® Movidius™ Neural Compute Stick, and FPGA—using a common API. [read more](https://docs.openvinotoolkit.org/)

## Support
If you have found this useful, you can donate by clicking on the [link ☘️](https://paypal.me/rydikov):

## Hardware Requirement

- Minimum Intel Gen 6 processors

## Installation OpenVINO on Raspbery Pi

Precompiled toolkit for Raspbian don't support ngraph and you must install OpenVINO manually.

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

## Installation Arugs
1. Clone project
```bash
git clone git@github.com:rydikov/argus.git
```
2. Install dependencies
```bash
cd argus
pip3 install -r requirements.txt
```

## Usage
### Run application with example config
```bash
$ python python run.py development.yml
```
### Config options
**sources**  (Required) 
*Set of data sources.*

**sources -> source-name**  (Required) 
*Source name.*

**sources -> source-name -> source**  (Required) 
*Source.*

**sources -> source-name -> stills_dir**  (Required) 
*Direcrory for saved frames.*

**sources -> source-name -> host_stills_uri**  (Required) 
*Web link to folder with frames.*

**sources -> source-name -> important_objects**  (Required) 
*Important objects. Mark an Alert if this objects detected on frame.*

**sources -> source-name -> other_objects**  
*Other objects. Mark if this objects detected on frame.*

**sources -> source-name -> max_object_area**  
*Max object area for detecton.*

**sources -> source-name -> save_every_n_frame**  
*Save every N frame.*

**sources -> source-name -> bfc**  
*Bad frame checker.*

Example:
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


## Credit

- AlexeyAB/darknet: [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
- [OpenVino](https://docs.openvinotoolkit.org/latest/index.html)
- https://stackoverflow.com/questions/66831806/loading-openvino-python-library-on-raspebrry-pi-4




---

cmake -DCMAKE_BUILD_TYPE=Release -DWITH_GSTREAMER=ON -DCMAKE_INSTALL_PREFIX=/usr/local ..

Copy python3 cv2 so
http://gstreamer-devel.966125.n4.nabble.com/RTSP-raw-RTP-video-frame-drops-td4677730.html

export PYTHONPATH=$PYTHONPATH:/PROJECT_PWD/argus
source /opt/intel/openvino_2021.3.394/bin/setupvars.sh
