#!/bin/bash
echo "ibmcloud login --sso -a cloud.ibm.com -r us-east"

ibmcloud ks cluster config --cluster c264ej5w0e9k1jc6fht0

kubectl apply -f deployment-cluster.yaml
kubectl apply -f deployment-ingress.yaml

kubectl scale -n default deployment simple-soe-deployment --replicas=0
kubectl scale -n default deployment simple-soe-deployment --replicas=2

echo "Pods:"
kubectl get pods

echo "Ingress"
kubectl describe ingress simple-soe-ingress

echo "get public IP from here:"
ibmcloud ks workers -c c264ej5w0e9k1jc6fht0
echo "get PORT (Nodeport) from here:"
kubectl describe service simple-soe-service

