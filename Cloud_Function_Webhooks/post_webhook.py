import json
import traceback
import random
import jwt

THANK_YOUS = ['Thank you', 'Gracias', 'Grazie', 'Merci', 'Danke sehr',
              'ありがとう', '谢谢你', 'जी शुक्रिया', 'σας ευχαριστώ', 'Спасибо']

def add_log_entry(comment, message=None):
    if message is None:
        print(comment)
    else:
        log_message = json.dumps(message)
        print(comment + " " + log_message)

def add_response_text_v2(message, text):
    if message['payload'].get('output') is not None:
        if message['payload']['output'].get('generic') is None:
            message['payload']['output']['generic'] = []
        message['payload']['output']['generic'].append({'response_type': 'text', 'text': text})
    return message

def get_thank_you():
    choice = int(random.random() * len(THANK_YOUS))
    return THANK_YOUS[choice]

def post_process_v2(message):
    text = get_thank_you()
    add_response_text_v2(message, text)
    return message

def main(request_input):
    add_log_entry('>>>>>>> Assistant post-webhook invoked')
    add_log_entry(request_input)
    try:
        auth = request_input['__ow_headers'].get('authorization')
        if auth is not None:
            jwt.decode(auth, 'webhook', algorithms=["HS256"])
        msg = post_process_v2(request_input)
        resp = {"body": msg}
    except Exception as e:
        add_log_entry('>>>>>>> Error: ' + str(e))
        add_log_entry(traceback.format_exc())
        resp = {"statusCode": 500, "body": str(e)}
    add_log_entry(resp)
    return resp

if __name__ == '__main__':
    main({"type": "post-webhook"})
