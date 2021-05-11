import json
import traceback
import random

insulting_names = ['Fart face', 'Dick nose', 'Butt head', 'Dumb ass', 'Bone head',
                   'Dip shit', 'Shit head']

def get_insulting_name():
    choice = int(random.random() * len(insulting_names))
    return insulting_names[choice]

def addLogEntry(comment, message=None):
    if message is None:
        print(comment)
    else:
        logmessage = json.dumps(message, separators=(',', ':'))
        print(comment + " " + logmessage)

def set_response_text(message, list_of_texts):
    if message.get('output') is not None:
        message['output']['text'] = []
        message['output']['generic'] = []
        for text in list_of_texts:
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


def post_process_v2(message):
    insulting_name = get_insulting_name()
    add_response_text_v2(message, [insulting_name])
    return message

def main(request_input):
    addLogEntry('>>>>>>> Assistant post-webhook invoked')
    addLogEntry(request_input)
    try:
        resp_data = post_process_v2(request_input)
    except Exception as e:
        addLogEntry('>>>>>>> Error: ' + str(e))
        addLogEntry(traceback.format_exc())
        addLogEntry('Error Return to Gateway', request_input)
        set_response_text(request_input, ['Dang it!', 'Cloud Functions suck'])
    resp = {"body": request_input}
    addLogEntry(resp)
    return resp

if __name__ == '__main__':
    main({"type": "post-webhook"})
