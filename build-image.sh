#!/bin/bash
echo "Be sure to start docker.app first"

docker rmi kineticsquid/sudoku-bot:latest
docker build --rm --no-cache --pull -t kineticsquid/sudoku-bot:latest .
docker push kineticsquid/sudoku-bot:latest

# list the current images
echo "Docker Images..."
docker images