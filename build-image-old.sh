#!/bin/bash
echo "Be sure to start docker.app first"

docker rmi kineticsquid/simple-soe:latest
docker build --rm --no-cache --pull -t kineticsquid/simple-soe:latest .
docker push kineticsquid/simple-soe:latest

# list the current images
echo "Docker Images..."
docker images