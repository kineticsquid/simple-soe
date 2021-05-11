#!/bin/bash

curl -X POST -u "apikey:${API_KEY}" --header "Content-Type:application/json" --data "{\"input\": {\"text\": \"Good morning CF!\"}}" "https://api.us-south.assistant.watson.cloud.ibm.com/instances/d178072a-20a1-43ec-8404-22721fead775/v2/assistants/8073b108-62d8-4244-b828-f5f733b40859/message?version=2020-04-01"