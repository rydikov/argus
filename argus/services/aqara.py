import hashlib
import json
import logging
import os.path
import requests
import time
import uuid

BASE_URL = 'https://open-ru.aqara.com/v3.0/open/api'

INCORRECT_REFRESH_TOKEN_CODE = 2006

logger = logging.getLogger('json')


class GetTokensError(Exception):
    pass


class AqaraService:
    def __init__(self, aqara_config, state_dir):
        self.app_id = aqara_config['app_id']
        self.app_key = aqara_config['app_key']
        self.key_id = aqara_config['key_id']
        self.scene_id = aqara_config['scene_id']
        self.account = aqara_config['account']
        
        self.access_token = None
        self.refresh_token = None
        self.expiresIn = None

        self.tokens_file_path = os.path.join(state_dir, 'tokens.json')
        self.code_file_path  = os.path.join(state_dir, 'code.json')

        if os.path.isfile(self.tokens_file_path):
            self._load_tokens()

    def _get_code(self):
        logger.info(f'Get code')
        data = {
            'intent': 'config.auth.getAuthCode',
            'data': {
                'account': self.account,
                'accountType': 0,
            }
        }
        resp = requests.post(BASE_URL, headers=self._get_headers(), json=data).json()['result']
        logger.info(f'Get code. Resp {resp}')

    def _get_tokens(self):
        
        try:
            with open(self.code_file_path, 'r') as f:
                code = json.load(f)['code']
        except Exception:
            raise GetTokensError('Code is None')
        else:
            os.remove(self.code_file_path)
       
        logger.info(f'Get tokens. Code is {code}')
        data = {
            'intent': 'config.auth.getToken',
            'data': {
                'authCode': code,
                'account': self.account,
                'accountType': 0
                }
            }
        resp = requests.post(BASE_URL, headers=self._get_headers(), json=data).json()
        if 'accessToken' in resp['result']:
            logger.info(f'Get tokens. Resp {resp}')
            self._save_tokens(resp['result'])
        else:
            logger.error(f'Get tokens. Error  {resp}')
            raise GetTokensError('Api error')

    def _load_tokens(self):
        logger.info(f'Load tokens')

        with open(self.tokens_file_path, 'r') as f:
            tokens = json.load(f)

        self.access_token = tokens['accessToken']
        self.refresh_token = tokens['refreshToken']
        self.expiresIn = int(tokens['expiresIn'])

    def save_code(self, code):
        logger.info(f'Save {code} to {self.code_file_path}')
        with open(self.code_file_path,  'w+') as f:
            json.dump({'code': code}, f, indent=4)

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
        resp = requests.post(BASE_URL, headers=self._get_headers(), json=data).json()

        if resp['code'] == INCORRECT_REFRESH_TOKEN_CODE:
            logger.error(f'Refresh tokens. Error {resp}')
            self.access_token = None
            raise GetTokensError('Api error')

        logger.info(f'Refresh tokens. Resp {resp}')
        self._save_tokens(resp['result'])

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

        if os.path.isfile(self.code_file_path):
            self._get_tokens()
        elif self.access_token is None:
            self._get_code()
        elif self.expiresIn and self.expiresIn < time.time():
            self._refresh_tokens()
        else:
            raise GetTokensError('No way')

        headers = self._get_headers(self.access_token)
        logger.info(f'Make request with headers {headers}')

        resp = requests.post(BASE_URL, headers=headers, json=data).json()

        logger.info(resp)
        return resp
    
    def run_scene(self):
        logger.info(f'Run scene')

        data = {
            'intent': 'config.scene.run',
            'data': {
                'sceneId': self.scene_id
            }
        }
        resp = self._make_request(data)
