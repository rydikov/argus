## Установка на хостовую машину

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

## Пример настройки Nginx для доступа к сохраненным кадрам
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

## Пример настройки Grafana для логов производительности и температуры

![Grafana](https://github.com/rydikov/argus/raw/main/res/grafana.png)

Loki config
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