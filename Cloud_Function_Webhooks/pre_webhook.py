import json
import traceback
import re
import jwt

def add_log_entry(comment, message=None):
    if message is None:
        print(comment)
    else:
        log_message = json.dumps(message)
        print(comment + " " + log_message)

def redact_input(text):
    redacted_text = re.sub(r'\d', '1', text)
    return redacted_text

def pre_process_v2(message):
    input = message['payload'].get('input')
    if input is not None:
        redacted_text = redact_input(input['text'])
        input['text'] = redacted_text
    return message

def main(request_input):
    add_log_entry('>>>>>>> Assistant pre-webhook invoked')
    add_log_entry(request_input)
    try:
        auth = request_input['__ow_headers'].get('authorization')
        if auth is not None:
            jwt.decode(auth, 'webhook', algorithms=["HS256"])
        msg = pre_process_v2(request_input)
        resp = {"body": msg}
    except Exception as e:
        add_log_entry('>>>>>>> Error: ' + str(e))
        add_log_entry(traceback.format_exc())
        resp = {"statusCode": 500, "body": str(e)}
    add_log_entry(resp)
    return resp

if __name__ == '__main__':
    main({"type": "pre-webhook"})
