#!/bin/bash

echo 'Be sure to "gcloud auth login" first'

export DATE=`date '+%F_%H:%M:%S'`

# Run this to create or re-deploy the function
gcloud run deploy sudoku-bot --allow-unauthenticated --project cloud-run-stuff --region us-central1 \
  --cpu=4 --memory=4Gi \
  --source ./ --set-env-vars=DATE=$DATE \
  --set-env-vars=TWILIO_ACCOUNT_SID=${TWILIO_ACCOUNT_SID} \
  --set-env-vars=TWILIO_AUTH_TOKEN=${TWILIO_AUTH_TOKEN} \
  --set-env-vars=REDIS_HOST=${REDIS_HOST} \
  --set-env-vars=REDIS_PORT=${REDIS_PORT} \
  --set-env-vars=REDIS_PW=${REDIS_PW} \
  --set-env-vars=SUDOKU_SOLVER_URL=${SUDOKU_SOLVER_URL} \
  --set-env-vars=JWT_SECRET=${JWT_SECRET}