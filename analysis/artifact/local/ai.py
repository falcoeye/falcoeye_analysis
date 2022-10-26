import json
import requests

import logging
from tensorflow_serving.apis import predict_pb2
import numpy as np
import tensorflow as tf

SERVERS = {}

def get_service_server(model_name,model_version,protocol,run_if_down):
	if model_name in SERVERS:
		return SERVERS[model_name]
	logging.info(f"Requesting {model_name} info")
	if protocol.lower() == "restful":
		SERVERS[model_name] = TensorflowServingRESTFul(model_name,model_version,'http://localhost:8501')
	elif protocol.lower() == "grpc":
		SERVERS[model_name] = TensorflowServinggRPC(model_name,model_version,'localhost:8500')
	else:
		raise NotImplementedError
	return SERVERS[model_name]
	

class TensorflowServing:
	def __init__(self,
		model_name,
		model_version,
		service_address):
		self._name = model_name
		self._version = model_version
		self._service_address = service_address
		
	@property
	def service_address(self):
		return self._service_address

	def post(self):
		raise NotImplementedError
	
	async def post_async(self,session,frame):
		raise NotImplementedError

class TensorflowServingRESTFul(TensorflowServing):
	def __init__(self,
		model_name,
		model_version,
		service_address):
		TensorflowServing.__init__(self,model_name,
			model_version,service_address)
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
				detections = json.loads(responseText)['predictions'][0]
				return detections
		except Exception as e:
			raise
			logging.error(e)

class TensorflowServinggRPC(TensorflowServing):
	def __init__(self,
		model_name,
		model_version,
		service_address):
		TensorflowServing.__init__(self,model_name,
			model_version,service_address)
		logging.info(f"New gRPC tensorflow serving initialized for {model_name} on {service_address}")

	def post(self,frame):
		raise NotImplementedError
			
	async def post_async(self,stub,frame):
		try:
			request = predict_pb2.PredictRequest()
			request.model_spec.name = self._name
			request.model_spec.signature_name = 'serving_default'
			frame = np.expand_dims(frame, axis=0) 
			request.inputs['input_tensor'].CopyFrom(tf.make_tensor_proto(frame))

			logging.info(f"Posting new frame asynchronously to gRPC")
			result = await stub.Predict(request)
			logging.info("Prediction received")
			# To match resful response
			boxes = np.array(result.outputs['detection_boxes'].float_val).reshape((-1,4)).tolist()
			classes = result.outputs['detection_classes'].float_val
			scores = result.outputs['detection_scores'].float_val
			detections = {'detection_boxes': boxes,
				'detection_classes':classes,
				'detection_scores': scores}
			return detections
		except Exception as e:
			raise
			logging.error(e)

