#!/bin/bash

# Run this once to create the Google Cloud project which will hold the lambda functions.
#
# After the project is created, need to link to a billing account. Otherwise the deploy will fail.
# Do this on the project page in the console by selecting 'Billing' on the left nav
#
# Re enabling the APIs, to see enabled, execute 'gcloud services list --project my-cloud-run-stuff'
# To see available APIs that can be enabled, execute 'gcloud services list --project my-cloud-run-stuff --available'

echo 'Be sure to "gcloud auth login" first'

gcloud projects create cloud-run-stuff --name cloud-run-stuff --enable-cloud-apis
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudfunctions.googleapis.com \
  cloudbuild.googleapis.com --project cloud-run-stuff
gcloud projects list
gcloud projects describe cloud-run-stuff