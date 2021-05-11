# simple-soe for Voice Agent

Cluster: `simple-soe-cluster-074b55ec662880a9b91b986213323a0b-0000.us-east.containers.appdomain.cloud`

`https://ibm.quip.com/BSLPABbbzpAB/SOE-Migration-Notes`

To get entry point, get public IP from `ibmcloud ks workers -c c264ej5w0e9k1jc6fht0` and 
get port from 'Nodeport' value from `kubectl describe service simple-soe-service`. This works for `http` only.
E.g.
  - `curl http://169.61.113.146:32453/`
  - `curl http://169.61.113.146:32453/build`
  
Ingress only works with `https`:
  - `https://simple-soe-cluster-074b55ec662880a9b91b986213323a0b-0000.us-east.containers.appdomain.cloud`
  - `https://simple-soe-cluster-074b55ec662880a9b91b986213323a0b-0000.us-east.containers.appdomain.cloud/`
  - `https://simple-soe-cluster-074b55ec662880a9b91b986213323a0b-0000.us-east.containers.appdomain.cloud/build`