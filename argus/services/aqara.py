import datetime
import hashlib
import json
import logging
import requests
import time
import uuid
import os.path

BASE_URL = 'https://open-ru.aqara.com/v3.0/open/api'

logger = logging.getLogger('json')


class AqaraService:
    def __init__(self, aqara_config):
        self.access_token = ''
        self.app_id = aqara_config['app_id']
        self.app_key = aqara_config['app_key']
        self.key_id = aqara_config['key_id']
        self.scene_id = aqara_config['scene_id']
        self.account = aqara_config['account']
        self.code = aqara_config.get('code')
        self.expiresIn = None


        self.tokens_file_path = '/tmp/tokens.json'
        if os.path.isfile(self.tokens_file_path):
            self._load_tokens()
            logger.info(f'Load tokens from file. Acces token  {self.access_token}')
        else:
            self._get_tokens()
            logger.info(f'Get tokens. Acces token  {self.access_token}')
        
    def _get_tokens(self):
        if self.code is None:
            logger.info(f'Get tokens. Code is None')
            data = {
                'intent': 'config.auth.getAuthCode',
                'data': {
                    'account': self.account,
                    'accountType': 0,
                }
            }
        else:
            logger.info(f'Get tokens. Code is {self.code}')
            data = {
	            'intent': 'config.auth.getToken',
	            'data': {
    	            'authCode': self.code,
    	            'account': self.account,
    	            'accountType': 0
                }
            }
        resp = self._make_request(data)['result']
        logger.info(f'Get tokens. Resp {resp}')
        if 'accessToken' in resp:
            self._save_tokens(resp)

    def _load_tokens(self):
        
        with open(self.tokens_file_path, 'r') as f:
            tokens = json.load(f)['result']

        self.access_token = tokens['accessToken']
        self.refresh_token = tokens['refreshToken']
        self.expiresIn = int(tokens['expiresIn'])

    def _save_tokens(self, tokens):
        self.access_token = tokens['accessToken']
        self.refresh_token = tokens['refreshToken']
        tokens['expiresIn'] = round(time.time()) + int(tokens['expiresIn'])
        self.expiresIn = tokens['expiresIn']
        with open(self.tokens_file_path, 'w+') as f:
            json.dump(tokens, f, indent=4)

    def _refresh_tokens(self):
        data = {
            'intent': 'config.auth.refreshToken',
            'data': {
                'refreshToken': self.refresh_token
            }
        }
        resp = self._make_request(data)['result']
        logger.info(f'Refresh tokens. Resp {resp}')
        self._save_tokens(resp)

    def _get_headers(self, access_token=''):

        nonce = str(uuid.uuid4()).split('-')[0]
        header_time = str(int(time.time()*1000))

        pre_sign = f'Appid={self.app_id}&Keyid={self.key_id}&Nonce={nonce}&Time={header_time}{self.app_key}'

        if access_token:
            pre_sign = f'Accesstoken={access_token}&{pre_sign}'

        sign = hashlib.md5(pre_sign.lower().encode()).hexdigest()

        return {
            'Accesstoken': access_token,
            'Appid': self.app_id,
            'Keyid': self.key_id,
            'Nonce': nonce,
            'Time': header_time,
            'Sign': sign
        } 
    
    def _make_request(self, data):
        
        if self.expiresIn and self.expiresIn < time.time():
            self._refresh_tokens()

        headers = self._get_headers(self.access_token)
        logger.info(f'Make request with headers {headers}')

        resp = requests.post(
            BASE_URL, 
            headers=headers, 
            json=data
        ).json()
        
        logger.info(resp)
        return resp
    
    def run_scene(self):
        data = {
            'intent': 'config.scene.run',
            'data': {
                'sceneId': self.scene_id
            }
        }
        resp = self._make_request(data)
