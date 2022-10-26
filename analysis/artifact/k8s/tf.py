import json
import requests
from .core import FalcoServingKube

import logging
from tensorflow_serving.apis import predict_pb2
import numpy as np
import tensorflow as tf
import time
import os


def start_tfserving(model_name, model_version,port,protocol):
	logging.info(f"Starting tensorflow serving on port {port} using {protocol} protocol")
	if protocol.lower() == "restful":
		target_port = 8501
	elif protocol.lower() == "grpc":	
		target_port = 8500
	
	kube = FalcoServingKube(model_name,
			port=port,
			targetport=target_port,
			template_type="tf",
			ready_message ="Entering the event loop",
			error_message =None)

	started = kube.start() and kube.is_running()
	logging.info(f"kube started for {model_name}?: {started}")
	if started:
		logging.info(f"New container for {model_name} started")
		logging.info(f"Waiting for container for {model_name} to load model")
		# TODO: fix, this is not safe
		while not kube.is_ready() or kube.did_fail():
			time.sleep(1)

		if kube.did_fail():
			kube.delete_deployment()
			kube.delete_service()
			logging.info(f"Failed to launch the kube for {model_name} for 3 minutes")
			return None


		logging.info(f"Container for {model_name} is ready to serve")
		deployment = os.getenv("DEPLOYMENT","local")
		service_address = kube.get_service_address(external=deployment=="local",
			hostname=deployment=="local")


		if protocol.lower() == "restful":
			tfserver = TensorflowServingRESTFul(model_name,model_version,service_address,kube)
		elif protocol.lower() == "grpc":	
			tfserver = TensorflowServinggRPC(model_name,model_version,service_address,kube)
		return tfserver
	else:
		logging.error(f"Couldn't start container for {model_name}")
		return None

class TensorflowServing:
	def __init__(self,
		model_name,
		model_version,
		service_address,
		kube):
		self._name = model_name
		self._version = model_version
		self._kube = kube
		self._service_address = service_address
		logging.info(f"New tensorflow serving container initialized for {model_name} on {service_address}")

	@property
	def service_address(self):
		return self._service_address

	def post(self):
		raise NotImplementedError
	
	async def post_async(self,session,frame):
		raise NotImplementedError
	
	def is_running(self):
		return self._kube.is_running()

class TensorflowServingRESTFul(TensorflowServing):
	def __init__(self,
		model_name,
		model_version,
		service_address,
		kube):
		TensorflowServing.__init__(self,model_name,
			model_version,service_address,kube)
		self._predict_url = f"{service_address}/v{model_version}/models/{model_name}:predict"
		logging.info(f"New RESTful tensorflow serving initialized for {model_name} on {service_address} {self._predict_url}")

	def post(self,frame):
		data = json.dumps({"signature_name": "serving_default", "instances": [frame.tolist()]})
		headers = {"content-type": "application/json"}
		response = requests.post(self._predict_url, data=data, headers=headers)
		detections = json.loads(response.text)['predictions'][0]
		return detections

	async def post_async(self,session,frame):
		try:
			data = json.dumps({"signature_name": "serving_default", "instances": [frame.tolist()]})
			headers = {"content-type": "application/json"}
			logging.info(f"Posting new frame asynchronously {self._predict_url}")
			async with session.post(self._predict_url,data=data,headers=headers) as response:
				responseText = await response.text()
				logging.info("Prediction received")
				detections = json.loads(responseText)
				return detections
		except Exception as e:
			raise
			logging.error(e)

class TensorflowServinggRPC(TensorflowServing):
	def __init__(self,
		model_name,
		model_version,
		service_address,
		kube):
		TensorflowServing.__init__(self,model_name,
			model_version,service_address,kube)
		
		logging.info(f"New gRPC tensorflow serving initialized for {model_name} on {service_address}")

	def post(self,frame):
		raise NotImplementedError
			
	async def post_async(self,stub,frame):
		try:
			logging.info(f"Posting new frame asynchronously to gRPC")
			request = predict_pb2.PredictRequest()
			logging.info(f"Request to gRPC created {self._name}")
			request.model_spec.name = self._name
			request.model_spec.signature_name = 'serving_default'
			logging.info(f"Expanding frame")
			frame = np.expand_dims(frame, axis=0) 
			request.inputs['input_tensor'].CopyFrom(tf.make_tensor_proto(frame))
			logging.info(f"Predicting")
			result = await stub.Predict(request)
			logging.info("Prediction received")
			
			return result
		except Exception as e:
			logging.error(e)
			return None