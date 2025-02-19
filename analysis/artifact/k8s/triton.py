import json
import logging
import numpy as np
import os
import time
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import InferenceServerException
from tritonclient.grpc import aio as triton_aio
from .core import FalcoServingKube

def start_tritonserving(model_name, model_version, port, protocol):
    logging.info(f"Starting Triton server on port {port} using {protocol} protocol")
    if protocol.lower() == "restful":
        target_port = 8000
    elif protocol.lower() == "grpc":
        target_port = 8001
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")

    kube = FalcoServingKube(model_name,
        port=port,
        targetport=target_port,
        template_type="triton",
        ready_message="Started Metrics Service",  # Triton's ready message
        error_message=None)
            
    started = kube.start() and kube.is_running()
    logging.info(f"kube started for {model_name}?: {started}")
    if started:
        logging.info(f"New container for {model_name} started")
        logging.info(f"Waiting for container for {model_name} to load model")

    # Wait for server to be ready
    while not kube.is_ready() or kube.did_fail():
        time.sleep(1)

    if kube.did_fail():
        kube.delete_deployment()
        kube.delete_service()
        logging.info(f"Failed to launch the kube for {model_name} for 3 minutes")
        return None

    logging.info(f"Container for {model_name} is ready to serve")
    if os.getenv("DEPLOYMENT") == "local":
        logging.info(f"Getting service address for local request")
        service_address = kube.get_service_address(external=True, hostname=True,port_name="grpc")
    else:
        logging.info(f"Getting service address for within-cluster request")
        service_address = kube.get_service_address(port_name="grpc")
    
    if protocol.lower() == "restful":
        raise NotImplementedError
    elif protocol.lower() == "grpc":
        triton_server = TritonServinggRPC(model_name, model_version, service_address, kube)
        return triton_server
    else:
        logging.error(f"Couldn't start container for {model_name}")
        return None

class TritonServing:
    def __init__(self,
                 model_name,
                 model_version,
                 service_address,
                 kube):
        self._name = model_name
        self._version = model_version
        self._kube = kube
        self._service_address = service_address
        logging.info(f"New Triton serving container initialized for {model_name} on {service_address}")

    @property
    def service_address(self):
        return self._service_address

    def post(self, client, inputs, outputs):
        raise NotImplementedError
    
    async def post_async(self, client, inputs, outputs):
        raise NotImplementedError
    
    def is_running(self):
        return self._kube.is_running()

class TritonServinggRPC(TritonServing):
    def __init__(self,
                 model_name,
                 model_version,
                 service_address,
                 kube):
        TritonServing.__init__(self, model_name,
                              model_version,
                              service_address,
                              kube)
        
        logging.info(f"New gRPC Triton serving initialized for {model_name} on {service_address}")

    def post(self, client, inputs, outputs):
        """
        Send synchronous inference request to Triton server
        
        Args:
            client: Triton client instance
            inputs: List of InferInput objects
            outputs: List of InferRequestedOutput objects
            
        Returns:
            Inference results dictionary
        """
        try:
            results = client.infer(
                model_name=self._name,
                inputs=inputs,
                outputs=outputs
            )
            return results
        except InferenceServerException as e:
            logging.error(f"Inference failed: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error during inference: {str(e)}")
            return None
            
    async def post_async(self, client, inputs, outputs):
        """
        Send asynchronous inference request to Triton server
        
        Args:
            client: Triton async client instance
            inputs: List of InferInput objects
            outputs: List of InferRequestedOutput objects
            
        Returns:
            Inference results dictionary
        """
        try:
            results = await client.infer(
                model_name=self._name,
                inputs=inputs,
                outputs=outputs
            )
            return results
        except Exception as e:
            logging.error(f"Error in async inference: {str(e)}")
            return None

def get_service_server(model_name, model_version, protocol, run_if_down=True):
    """Helper function to get or start Triton server"""
    logging.info(f"Getting model server for {model_name}")
    deployment = os.getenv("DEPLOYMENT", "local")

    if deployment == "k8s":
        port = 8001 if protocol.lower() == "grpc" else 8000
        service = start_tritonserving(model_name, model_version, port, protocol)
        return service
    else:
        raise NotImplementedError("Local deployment not implemented for Triton")