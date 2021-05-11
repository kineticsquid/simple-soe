#!/bin/bash

ibmcloud target -r us-east -g "JKs Resource Group"
ibmcloud fn namespace target Kellerman-Functions

ibmcloud fn action update utils/pre-webhook ./pre_webhook.py --kind python:3.7
ibmcloud fn action update utils/post-webhook ./post_webhook.py --kind python:3.7


# Get the definition of the functions
ibmcloud fn action get utils/pre-webhook
ibmcloud fn action get utils/post-webhook


