#!/bin/bash
echo "Be sure to start docker.app first"

docker rmi kineticsquid/simple-soe-base:latest
docker build --rm --no-cache --pull -t kineticsquid/simple-soe-base:latest -f Dockerfile-base .
docker push kineticsquid/simple-soe-base:latest

# list the current images
echo "Docker Images..."
docker images