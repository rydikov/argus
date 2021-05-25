# argus

https://stackoverflow.com/questions/66831806/loading-openvino-python-library-on-raspebrry-pi-4

cmake -DCMAKE_BUILD_TYPE=Release -DWITH_GSTREAMER=ON -DCMAKE_INSTALL_PREFIX=/usr/local ..

Copy python3 cv2 so

export PYTHONPATH=$PYTHONPATH:/PROJECT_PWD/argus
source /opt/intel/openvino_2021.3.394/bin/setupvars.sh
