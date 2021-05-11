#!/bin/bash

echo "http://0.0.0.0:5005/build"

# Now run locally. Use "rm" to remove the container once it finishes
docker run --rm -p 5005:5040 --env ASSISTANT_URL=${ASSISTANT_URL} --env API_KEY=${API_KEY} --env API_VERSION=${API_VERSION} --env PYTHONUNBUFFERED=1 kineticsquid/simple-soe:latest

#docker run --rm -p 5005:5040 kineticsquid/simple-soe:latest



