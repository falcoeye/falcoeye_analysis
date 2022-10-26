
from kubernetes import client, config
import sys

config.load_kube_config()
api_instance = client.CoreV1Api()
# TODO: rename this
def get_ip_address(service_name):
    services = api_instance.list_namespaced_service("default")
    for s in services.items:
        label = s.metadata.labels["run"]
        if label == f'{service_name}-service':
            return s.status.load_balancer.ingress[0].hostname
    return None