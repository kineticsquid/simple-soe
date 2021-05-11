import os
import sys
import json
from flask import Flask, request, jsonify
import traceback
import random
import requests
import re

PORT = os.getenv('PORT', '5040')
ASSISTANT_URL = os.environ['ASSISTANT_URL']
API_KEY = os.environ['API_KEY']
API_VERSION = os.environ['API_VERSION']
CUSTOMER_NAMES = ['Sydney', 'Carol', 'Alex', 'Blake', 'Kyle', 'Taylor', 'Jordan']

app = Flask(__name__)

@app.before_request
def do_something_whenever_a_request_comes_in():
    r = request
    addLogEntry('>>>>>>>>>>>>>>>>>>>> %s %s' % (r.method, r.url))
    addLogEntry(r.headers)
    data = r.data
    if len(data) > 0:
        addLogEntry(json.loads(r.data))


@app.errorhandler(Exception)
def handle_bad_request(e):
    addLogEntry('>>>>>>> Error: ' + str(e))
    addLogEntry(traceback.format_exc())
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
    addLogEntry(msg)
    return jsonify(msg)

@app.route('/pre-webhook', methods=['POST'])
def pre_webhook():
    data = request.data
    msg = json.loads(data)
    addLogEntry('>>>>>>> Assistant pre-webhook invoked')
    msg = pre_process_v2(msg)
    addLogEntry(msg)
    response = jsonify(msg)
    return response

@app.route('/post-webhook', methods=['POST'])
def post_webhook():
    data = request.data
    msg = json.loads(data)
    addLogEntry('>>>>>>> Assistant post-webhook invoked')
    msg = post_process_v2(msg)
    addLogEntry(msg)
    response = jsonify(msg)
    return response

def handle_message_api(message):
    try:
        input = message['input'].get('text')
        addLogEntry('>>>>>>> New \message call input: \'%s\'' % input)
        message, customer_name = pre_process(message)
        addLogEntry(">>>>>>> Pre-processed input sent to Assistant: \'%s\'" % message['input']['text'])
        message = call_assistant(message)
        output = message['output'].get('text')
        addLogEntry(">>>>>>> Output back from Assistant: \'%s\'" % output)
        message = post_process(message, customer_name)
        addLogEntry(">>>>>>> Output after post processing: \'%s\'" % message['output']['text'])
    except Exception as e:
        addLogEntry('>>>>>>> Error: ' + str(e))
        addLogEntry(traceback.format_exc())
    return message


def pre_process(message):
    customer_name = get_customers_name()
    input = message.get('input')
    if input is not None:
        redacted_text = re.sub(r'\d', '9', input['text'])
        input['text'] = redacted_text
    return message, customer_name

def pre_process_v2(message):
    customer_name = get_customers_name()
    input = message['payload'].get('input')
    if input is not None:
        redacted_text = re.sub(r'\d', '9', input['text'])
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


def post_process(message, customer_name):
    add_response_text(message, customer_name)
    return message

def post_process_v2(message):
    insulting_name = get_insulting_name()
    add_response_text_v2(message, [insulting_name])
    return message


def addLogEntry(comment, message=None):
    if message is None:
        print(comment)
    else:
        logmessage = json.dumps(message)
        print(comment + " " + logmessage)


def get_customers_name():
    choice = int(random.random() * len(CUSTOMER_NAMES))
    return CUSTOMER_NAMES[choice]


def add_response_text(message, text):
    if message.get('output') is not None:
        if message['output'].get('text') is not None and message['output'].get('generic') is not None:
            message['output']['text'].append(text)
            message['output']['generic'].append({'response_type': 'text', 'text': text})
    return message

def add_response_text_v2(message, list_of_texts):
    if message['payload'].get('output') is not None:
        if message['payload']['output'].get('generic') is None:
            message['payload']['output']['generic'] = []
        for text in list_of_texts:
            message['payload']['output']['generic'].append({'response_type': 'text', 'text': text})
    return message


print('Starting %s....' % sys.argv[0])
print('Python: ' + sys.version)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(PORT))
