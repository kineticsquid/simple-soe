#!/bin/bash
echo "ibmcloud login --sso"

ibmcloud target -r us-south -g "JKs Resource Group"
ibmcloud ce project select --name "Simple SOE - Serverless"
ibmcloud ce application update --name simple-soe-app-04 --image docker.io/kineticsquid/simple-soe --env JWT_SECRET=${JWT_SECRET} --env ASSISTANT_URL=${ASSISTANT_URL} --env API_KEY=${API_KEY} --env API_VERSION=${API_VERSION}
ibmcloud ce application get --name simple-soe-app-04
ibmcloud ce application events --application simple-soe-app-04
ibmcloud ce application logs --app simple-soe-app-04
