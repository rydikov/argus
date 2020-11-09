import requests


def send_message(message):

    bot_token = ***REMOVED***
    bot_chat_id = '53757246'
    dsn_template = 'https://api.telegram.org/bot{}/sendMessage?chat_id={}&parse_mode=Markdown&text={}'

    send_text = dsn_template.format(
        bot_token,
        bot_chat_id,
        message
    )

    response = requests.get(send_text)

    return response.json()
