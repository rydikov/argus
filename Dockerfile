FROM openvino/ubuntu24_runtime:2025.4.1

WORKDIR /app
USER root

COPY models models
COPY res res

COPY requires.txt requires.txt
RUN python -m pip install --upgrade pip
RUN pip install -r requires.txt

COPY argus argus
COPY setup.py setup.py
RUN pip install -e .
