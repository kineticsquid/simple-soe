import requests
import cv2
import os
import numpy as np

# API_KEY = os.environ['API_KEY']
API_VERSION = '2020-04-01'
http_headers = {'content-type': 'application/json'}
ASSISTANT_URL = 'https://api.us-south.assistant.watson.cloud.ibm.com/instances/df8d9ee2-3c04-44b7-81f1-fb2eae978e56/v2/assistants/a136bc55-8a3f-4b0b-98e0-852ea06f4d98/sessions'
WEBHOOK_URL ='http://0.0.0.0:5040/webhook'

message = {
  "event": {
    "name": "message_received"
  },
  "options": {},
  "payload": {
    "input": {
      "message_type": "text",
      "text": "Hi",
      "source": {
        "type": "user",
        "id": "2dd391dab0f09603c67f455ab6ab435b"
      },
      "options": {
        "suggestion_only": False,
        "return_context": True
      }
    },
    "context": {
      "global": {
        "system": {
          "user_id": "2dd391dab0f09603c67f455ab6ab435b"
        }
      },
      "skills": {
        "main skill": {
          "user_defined": {
            "smsTenantPhoneNumber": "+12183003720",
            "smsUserPhoneNumber": "+19192446142",
            "smsSessionID": "text_messaging-2dd391dab0f09603c67f455ab6ab435b",
            "smsMedia": []
          }
        }
      }
    }
  }
}

puzzles = ['https://i.imgur.com/NnnZcYf.png',
           'https://i.imgur.com/Fum8e7m.jpg',
           'https://i.imgur.com/HXzhPo4.jpg']

for p in puzzles:
  response = requests.get(p)
  if response.status_code == 200:
      results = response.content
      image_bytearray = np.asarray(bytearray(results), dtype="uint8")
      puzzle_image = cv2.imdecode(image_bytearray, cv2.IMREAD_COLOR)
      height, width, channels = puzzle_image.shape
      print(p)
      print("\tlength bytearray: %s" % len(image_bytearray))
      print("\timage dimensions: %s x %s x %s" % (height, width, channels))

      if height > 1280:
        reduction = int(np.log2(height/1280)) + 1
        print(reduction)
        new_dim = (int(width/2**reduction), int(height/2**reduction))
        resized_image = cv2.resize(puzzle_image, new_dim)
        height, width, channels = resized_image.shape
        print("\timage dimensions: %s x %s x %s" % (height, width, channels))





