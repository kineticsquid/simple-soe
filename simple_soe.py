import os
import sys
import json
from flask import Flask, request, jsonify
import traceback
import random
import requests
import re
import jwt

PORT = os.getenv('PORT', '5040')
ASSISTANT_URL = os.environ['ASSISTANT_URL']
API_KEY = os.environ['API_KEY']
API_VERSION = os.environ['API_VERSION']
JWT_SECRET = os.environ['JWT_SECRET']
THANK_YOUS = ['Thank you', 'Gracias', 'Grazie', 'Merci', 'Danke sehr',
              'ありがとう', '谢谢你', 'जी शुक्रिया', 'σας ευχαριστώ', 'Спасибо']

app = Flask(__name__)

@app.before_request
def do_something_whenever_a_request_comes_in():
    r = request
    add_log_entry('>>>>>>>>>>>>>>>>>>>> %s %s' % (r.method, r.url))
    data = r.data
    if len(data) > 0:
        add_log_entry(json.loads(r.data))
    auth = r.headers.get('authorization')
    if auth is not None:
        jwt.decode(auth, JWT_SECRET, algorithms=["HS256"])


@app.errorhandler(Exception)
def handle_bad_request(e):
    add_log_entry('>>>>>>>>>>>>>>>>>>>> Error: ' + str(e))
    add_log_entry(traceback.format_exc())
    return str(e)


@app.route('/')
def index():
    return app.send_static_file('build.txt')


@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon-96x96.png')


@app.route('/message', methods=['POST'])
def message_api():
    msg = json.loads(request.data)
    msg = handle_message_api(msg)
    add_log_entry(msg)
    return jsonify(msg)

@app.route('/pre-webhook', methods=['POST'])
def pre_webhook():
    data = request.data
    msg = json.loads(data)
    add_log_entry('>>>>>>> Assistant pre-webhook invoked')
    msg = pre_process_v2(msg)
    add_log_entry(msg)
    response = jsonify(msg)
    return response

@app.route('/post-webhook', methods=['POST'])
def post_webhook():
    data = request.data
    msg = json.loads(data)
    add_log_entry('>>>>>>>>>>>>>>>>>>>> Assistant post-webhook invoked')
    msg = post_process_v2(msg)
    add_log_entry(msg)
    response = jsonify(msg)
    return response

def handle_message_api(message):
    try:
        input = message['input'].get('text')
        add_log_entry('>>>>>>>>>>>>>>>>>>>> New \message call input: \'%s\'' % input)
        message = pre_process(message)
        add_log_entry(">>>>>>>>>>>>>>>>>>>> Pre-processed input sent to Assistant: \'%s\'" % message['input']['text'])
        message = call_assistant(message)
        output = message['output'].get('text')
        add_log_entry(">>>>>>>>>>>>>>>>>>>> Output back from Assistant: \'%s\'" % output)
        message = post_process(message)
        add_log_entry(">>>>>>>>>>>>>>>>>>>> Output after post processing: \'%s\'" % message['output']['text'])
    except Exception as e:
        add_log_entry('>>>>>>>>>>>>>>>>>>>> Error: ' + str(e))
        add_log_entry(traceback.format_exc())
    return message


def pre_process(message):
    input = message.get('input')
    if input is not None:
        redacted_text = redact_input(input['text'])
        input['text'] = redacted_text
    return message

def redact_input(text):
    redacted_text = re.sub(r'\d', '1', text)
    return redacted_text

def pre_process_v2(message):
    input = message['payload'].get('input')
    if input is not None:
        redacted_text = redact_input(input['text'])
        input['text'] = redacted_text
    return message


def call_assistant(message):
    url = '%s?version=%s' % (ASSISTANT_URL, API_VERSION)
    http_headers = {'content-type': 'application/json'}
    result = requests.post(url, auth=('apikey', API_KEY), headers=http_headers, data=json.dumps(message))
    if result.status_code != 200:
        raise Exception('%s error calling Assistant.' % result.status_code)
    message = result.json()
    return message


def post_process(message):
    text = get_thank_you()
    add_response_text(message, text)
    return message

def post_process_v2(message):
    text = get_thank_you()
    add_response_text_v2(message, text)
    return message

def get_thank_you():
    choice = int(random.random() * len(THANK_YOUS))
    return THANK_YOUS[choice]


def add_log_entry(comment, message=None):
    if message is None:
        print(comment)
    else:
        log_message = json.dumps(message)
        print(comment + " " + log_message)


def add_response_text(message, text):
    if message.get('output') is not None:
        if message['output'].get('text') is not None and message['output'].get('generic') is not None:
            message['output']['text'].append(text)
            message['output']['generic'].append({'response_type': 'text', 'text': text})
    return message

def add_response_text_v2(message, text):
    if message['payload'].get('output') is not None:
        if message['payload']['output'].get('generic') is None:
            message['payload']['output']['generic'] = []
        message['payload']['output']['generic'].append({'response_type': 'text', 'text': text})
    return message


print('Starting %s....' % sys.argv[0])
print('Python: ' + sys.version)
build_file = open('./static/build.txt')
print('Running build:')
for line in build_file.readlines():
    print(line.strip())
build_file.close()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(PORT))
