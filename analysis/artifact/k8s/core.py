import json
import logging
import os

import yaml
from kubernetes import client, config, utils
import kubernetes

logger = logging.getLogger(__name__)

SERVING_TEMPLATE = {
    "tf":os.path.join(
        os.path.dirname(__file__), "resources/tf-serving-template.yml"
    ),
    "torch":os.path.join(
        os.path.dirname(__file__), "resources/torch-serving-template.yml"
    ),
    "job":os.path.join(
        os.path.dirname(__file__), "resources/analysis-job-template.yml"
    ), 
    "triton":os.path.join(
        os.path.dirname(__file__), "resources/triton-serving-template.yml"
    ),
}


def skip_if_already_exists(e):
    info = json.loads(e.api_exceptions[0].body)
    if info.get("reason").lower() == "alreadyexists":
        return True
    else:
        logger.debug(e)
        return False

class FalcoServingKube:
    ARTIFACT_REGISTRY = None
    def __init__(
        self,
        name,
        template_type=None,
        image=None,
        replicas=1,
        port=8501,
        targetport=8501,
        namespace="default",
        ready_message=None,
        error_message=None
    ):
        self.name = name
        self.service_name = self.name+"-svc"
        self.base_name = name.split("/")[-1]
        self.template_type = template_type
        self.image = image
        self.replicas = replicas
        self.port = port
        self.targetport = targetport
        self.namespace = namespace
        self.ready_message = ready_message
        self.error_message = error_message


        if not self.name:
            raise RuntimeError("name should not be empty")
        self.template = self._get_deployment_template()
   
        try:
            config.load_kube_config()
        except:
            config.load_incluster_config()

    def _get_deployment_template(self):
        
        if self.template_type is None:
            return None

        if isinstance(self.template_type, str) and self.template_type in SERVING_TEMPLATE:
            with open(SERVING_TEMPLATE[self.template_type]) as f:
                template = self._fill_deployment_template(f.read())
                template = list(yaml.safe_load_all(template))
        else:
            raise NotImplementedError(
                f"parsing template of type {type(self.template)} is not implemented yet"
            )

        return template

    def _fill_deployment_template(self, template):
        template = template.replace("$appname", self.name)
        template = template.replace("$replicas", str(self.replicas))
        template = template.replace("$port", str(self.port))
        template = template.replace("$targetport", str(self.targetport))
        
        # @jalalirs: create it if None but don't set it (i.e. self.image=...)
        # I don't like setting object attributes inside operation functions
        image = self.image
        if not image:
            image = f"{self.name}:latest"
            if FalcoServingKube.ARTIFACT_REGISTRY:
                image = f"{FalcoServingKube.ARTIFACT_REGISTRY}/{image}"
        
        template = template.replace("$image", image)

        return template

    def start(self):
        k8s_client = client.ApiClient()
        yaml_objects = self.template
        for data in yaml_objects:
            try:
                utils.create_from_dict(
                    k8s_client, data=data, namespace=self.namespace, verbose=True
                )

            except utils.FailToCreateError as e:
                return skip_if_already_exists(e)

        return True
    
    def is_ready(self):
        logs = self.get_logs()
        return logs and self.ready_message in logs
    
    def did_fail(self):
        logs = self.get_logs()
        return logs and (self.error_message and self.error_message in logs)
    
    def delete_deployment(self):
        api = client.AppsV1Api()
        try:
            api.delete_namespaced_deployment(
                name=self.name,
                namespace=self.namespace,
                body=client.V1DeleteOptions(
                    propagation_policy="Foreground", grace_period_seconds=5
                ),
            )
            logger.info(f"deployment `{self.name}` deleted.")
        except client.exceptions.ApiException as e:
            if e.reason == "Not Found":
                logger.debug(f"Deployment {self.name} has been deleted already.")

    def delete_service(self):
        api = client.CoreV1Api()
        try:
            api.delete_namespaced_service(
                name=self.name,
                namespace=self.namespace,
                body=client.V1DeleteOptions(
                    propagation_policy="Foreground", grace_period_seconds=5
                ),
            )
            logger.info(f"deployment `{self.name}` deleted.")
        except client.exceptions.ApiException as e:
            if e.reason == "Not Found":
                logger.debug(f"Deployment {self.name} has been deleted already.")

    def deployment_exists(self):
        v1 = client.AppsV1Api()
        resp = v1.list_namespaced_deployment(namespace=self.namespace)
        for i in resp.items:
            if i.metadata.name == self.name:
                return True
        return False

    def service_exists(self):
        v1 = client.CoreV1Api()
        resp = v1.list_namespaced_service(namespace=self.namespace)
        for i in resp.items:
            if i.metadata.name == self.name:
                return True
        return False

    def is_running(self):
        if self.deployment_exists() and self.service_exists():
            return True
        return False

    def get_service_address(self, external=False, hostname=False,port_name=None):
        if not self.is_running():
            logger.error(f"No running deployment found for {self.name}.")
            return None

        v1 = client.CoreV1Api()
        service = v1.read_namespaced_service(namespace=self.namespace, name=self.name)
        # If port_name is specified, find that specific port
        if port_name:
            
            port = next((port.port for port in service.spec.ports if getattr(port, 'name', None) == port_name), None)
            if port is None:
                logger.error(f"No port found with name {port_name}")
                return None
        else:
            # If no port_name specified, use the first portl
            logging.info(f"Service ports: {service.spec.ports} {self.name}")
            port = service.spec.ports[0].port
        
        if external:
            try:
                service = v1.read_namespaced_service(namespace=self.namespace, name=self.service_name)
            except Exception :
                # trying without -svc
                service = v1.read_namespaced_service(namespace=self.namespace, name=self.service_name[:-4])
            if hostname:
                host = service.status.load_balancer.ingress[0].hostname
            else:
                host = service.status.load_balancer.ingress[0].ip
        else:
            host = service.spec.cluster_ip

        return f"{host}:{port}"
    
    def get_logs(self,pod_name=None):
        v1 = client.CoreV1Api()
        _pod_name = None
        if not pod_name:
            api_response = v1.list_namespaced_pod(namespace=self.namespace)
            for item in api_response.items:
                item_podname = item.metadata.name
                
                if self.name in item_podname:
                    _pod_name = item_podname
                    break
        else:
            raise NotImplementedError

        if _pod_name:
            try:
                api_response = v1.read_namespaced_pod_log(name=_pod_name, namespace=self.namespace)
                return api_response
            except kubernetes.client.exceptions.ApiException as e:
                return None
        return None
        
    @staticmethod
    def set_artifact_registry(registry):
        if registry[-1] == "/":
            registry = registry[:-1]
        FalcoServingKube.ARTIFACT_REGISTRY = registry

    @staticmethod
    def is_port_taken(port,namespace="default"):
        try:
            config.load_kube_config()
        except:
            config.load_incluster_config()
        v1 = client.CoreV1Api()
        resp = v1.list_namespaced_service(namespace=namespace)
        ports = list(set([i.spec.ports[0].port for i in resp.items]))
        return port in ports

    def scale(self, n):
        pass

class FalcoJobKube:
    def __init__(
        self,
        name,
        job_path,
        namespace="default"
    ):
        self._name = name
        self._job_path = job_path
        self._namespace = namespace
        
        if not self._name:
            raise RuntimeError("name should not be empty")
        self._template = self._get_deployment_template()
   
        try:
            config.load_kube_config()
        except:
            config.load_incluster_config()
    
    def _get_deployment_template(self):
        
        with open(SERVING_TEMPLATE["job"]) as f:
            template = self._fill_deployment_template(f.read())
            template = list(yaml.safe_load_all(template))[0]

        return template

    def _fill_deployment_template(self, template):
        template = template.replace("$jobname", self._name)        
        template = template.replace("$analysis_path", self._job_path)
        logging.info(template)
        return template
    
    def start(self):
        k8s_client = client.V1Job()
        # Create the specification of deployment
        spec = client.V1JobSpec(
            template=self._template,
            backoff_limit=4)
        # Instantiate the job object
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(name=self._name),
            spec=spec)
        
        return job

