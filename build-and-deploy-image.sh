#!/bin/bash
echo "Be sure to start docker.app first"
echo "Building image..."

docker rmi kineticsquid/simple-soe:latest
docker build --rm --no-cache --pull -t kineticsquid/simple-soe:latest .
docker push kineticsquid/simple-soe:latest

# list the current images
echo "Getting docker Images..."
docker images

echo "Deploying image on code engine..."
ic target -r us-south -g default
ic ce proj select -n Utils

REV=$(date +"%y-%m-%d-%H-%M-%S")
echo ${REV}

ic ce app update -n simple-soe-app -i docker.io/kineticsquid/simple-soe:latest --min 1 --memory 8G --rn ${REV} --env JWT_SECRET=${JWT_SECRET} --env ASSISTANT_URL=${ASSISTANT_URL} --env API_KEY=${API_KEY} --env API_VERSION=${API_VERSION} --env SUDOKU_SOLVER_URL=${SUDOKU_SOLVER_URL} --env REDIS_HOST=${REDIS_HOST} --env REDIS_PORT=${REDIS_PORT} --env REDIS_PW=${REDIS_PW} --env TWILIO_ACCOUNT_SID=${TWILIO_ACCOUNT_SID} --env TWILIO_AUTH_TOKEN=${TWILIO_AUTH_TOKEN}
ic ce rev list --app simple-soe-app
ic ce app get -n simple-soe-app
#ic ce app events --app simple-soe-app
#ic ce app logs --app simple-soe-app

echo "webhook app endpoint:"
echo "https://simple-soe-app.9mzop27k89f.us-south.codeengine.appdomain.cloud"