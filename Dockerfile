FROM openvino/ubuntu20_runtime:2022.3.1

WORKDIR /app

COPY models models
COPY res res

COPY requires.txt requires.txt
RUN python3.8 -m pip install --upgrade pip
RUN pip install -r requires.txt

COPY argus argus
COPY setup.py setup.py
RUN pip install -e .
