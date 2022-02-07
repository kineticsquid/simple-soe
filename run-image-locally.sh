#!/bin/bash

docker pull kineticsquid/sudoku-bot:latest
# Now run locally. Use "rm" to remove the container once it finishes
docker run --rm -p 5020:5020 --env JWT_SECRET=${JWT_SECRET} \
  --env ASSISTANT_URL=${ASSISTANT_URL} \
  --env API_KEY=${API_KEY} \
  --env API_VERSION=${API_VERSION} \
  --env SUDOKU_SOLVER_URL=${SUDOKU_SOLVER_URL} \
  --env REDIS_HOST=${REDIS_HOST} \
  --env REDIS_PORT=${REDIS_PORT} \
  --env REDIS_PW=${REDIS_PW} \
  --env PORT=${PORT} \
  --env TWILIO_ACCOUNT_SID=${TWILIO_ACCOUNT_SID} \
  --env TWILIO_AUTH_TOKEN=${TWILIO_AUTH_TOKEN} \
  kineticsquid/sudoku-bot:latest

#docker run --rm -p 5005:5040 kineticsquid/simple-soe:latest



