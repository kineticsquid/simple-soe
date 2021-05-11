#!/bin/bash

curl -X POST -u "apikey:${API_KEY}" --header "Content-Type:application/json" --data "{\"input\": {\"text\": \"Good morning SOE!\"}}" "https://api.us-south.assistant.watson.cloud.ibm.com/instances/d178072a-20a1-43ec-8404-22721fead775/v2/assistants/f870c4ee-347d-4f46-91ee-a35282f24715/message?version=2020-04-01"