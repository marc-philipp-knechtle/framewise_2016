kubectl delete job framewise-2016-cluster

export NAME=ls6-stud-registry.informatik.uni-wuerzburg.de/extch1-framewise_2016:0.0.1

docker build -f cluster/Dockerfile . -t $NAME

docker login ls6-stud-registry.informatik.uni-wuerzburg.de
docker push $NAME

kubectl apply -f cluster/k8s.yaml

kubectl get pods --watch