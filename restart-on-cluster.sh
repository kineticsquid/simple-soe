#!/bin/bash
echo "ibmcloud login --sso -a cloud.ibm.com -r us-south"

ibmcloud ks cluster config --cluster c264ej5w0e9k1jc6fht0

kubectl scale -n default deployment simple-soe-deployment --replicas=0
kubectl scale -n default deployment simple-soe-deployment --replicas=2

echo "Pods:"
kubectl get pods

