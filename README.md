# Argus. Object detecton using OpenVINO and YOLO v4

| Details            |              |
|-----------------------|---------------|
| Programming Language: |  [![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/) |
| Intel OpenVINO ToolKit: |[![OpenVINO 2020.3](https://img.shields.io/badge/openvino-2020.3-blue.svg)](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html)|
| Hardware Used: | Raspberry Pi B+ |
| Device: | CPU or Intel Neural Cumpute Stick 2 or other Intel VPUs devices |



Argus application uses Deep Learning/Machine Learning to recognize objects on sources stream. Sources can be cameras and videos.
Application save frames with detection objecs and has the ability to telegam alert.

By utilizing pre-trained models and Intel OpenVINO toolkit with OpenCV. 

This application executes parallel threads for capture frames from sources and make async infer requests for objects detection.

**What is OpenVino?**

OpenVino (OpenVisual Inferencing and Neural Network Optimization) is toolkit to develop Deep Learning Application especially for Computer Vision by Intel. OpenVino Enables deep learning inference at the edge and supports heterogeneous execution across computer vision accelerators—CPU, GPU, Intel® Movidius™ Neural Compute Stick, and FPGA—using a common API. [read more](https://docs.openvinotoolkit.org/)

## Support
If you have found this useful, you can donate by clicking on the [link ☘️](https://paypal.me/rydikov):


### Tutorial

## Hardware Requirement

- Minimum Intel Gen 6 processors

## Installation

- Download and installOpenVINO 2020.3

- Download the converted YOLO v4 detection model.
```bash
wget 
wget 
```


## Usage

```bash
$ python python run.py development.yml
```

### Example Usage


## Credit

- AlexeyAB/darknet: [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
- [OpenVino](https://docs.openvinotoolkit.org/latest/index.html)




---
https://stackoverflow.com/questions/66831806/loading-openvino-python-library-on-raspebrry-pi-4

cmake -DCMAKE_BUILD_TYPE=Release -DWITH_GSTREAMER=ON -DCMAKE_INSTALL_PREFIX=/usr/local ..

Copy python3 cv2 so
http://gstreamer-devel.966125.n4.nabble.com/RTSP-raw-RTP-video-frame-drops-td4677730.html

export PYTHONPATH=$PYTHONPATH:/PROJECT_PWD/argus
source /opt/intel/openvino_2021.3.394/bin/setupvars.sh
