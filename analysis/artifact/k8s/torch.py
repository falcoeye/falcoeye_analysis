import json
import requests
from .core import FalcoServingKube
import logging
import numpy as np
import grpc
import io
from grpc import aio 
import asyncio
import aiohttp 
import time
from PIL import Image

"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0finference.proto\x12 org.pytorch.serve.grpc.inference\x1a\x1bgoogle/protobuf/empty.proto\"\xbd\x01\n\x12PredictionsRequest\x12\x12\n\nmodel_name\x18\x01 \x01(\t\x12\x15\n\rmodel_version\x18\x02 \x01(\t\x12N\n\x05input\x18\x03 \x03(\x0b\x32?.org.pytorch.serve.grpc.inference.PredictionsRequest.InputEntry\x1a,\n\nInputEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x0c:\x02\x38\x01\"(\n\x12PredictionResponse\x12\x12\n\nprediction\x18\x01 \x01(\x0c\"*\n\x18TorchServeHealthResponse\x12\x0e\n\x06health\x18\x01 \x01(\t2\xf1\x01\n\x14InferenceAPIsService\x12\\\n\x04Ping\x12\x16.google.protobuf.Empty\x1a:.org.pytorch.serve.grpc.inference.TorchServeHealthResponse\"\x00\x12{\n\x0bPredictions\x12\x34.org.pytorch.serve.grpc.inference.PredictionsRequest\x1a\x34.org.pytorch.serve.grpc.inference.PredictionResponse\"\x00\x42\x02P\x01\x62\x06proto3')



_PREDICTIONSREQUEST = DESCRIPTOR.message_types_by_name['PredictionsRequest']
_PREDICTIONSREQUEST_INPUTENTRY = _PREDICTIONSREQUEST.nested_types_by_name['InputEntry']
_PREDICTIONRESPONSE = DESCRIPTOR.message_types_by_name['PredictionResponse']
_TORCHSERVEHEALTHRESPONSE = DESCRIPTOR.message_types_by_name['TorchServeHealthResponse']
PredictionsRequest = _reflection.GeneratedProtocolMessageType('PredictionsRequest', (_message.Message,), {

  'InputEntry' : _reflection.GeneratedProtocolMessageType('InputEntry', (_message.Message,), {
	'DESCRIPTOR' : _PREDICTIONSREQUEST_INPUTENTRY,
	'__module__' : 'inference_pb2'
	# @@protoc_insertion_point(class_scope:org.pytorch.serve.grpc.inference.PredictionsRequest.InputEntry)
	})
  ,
  'DESCRIPTOR' : _PREDICTIONSREQUEST,
  '__module__' : 'inference_pb2'
  # @@protoc_insertion_point(class_scope:org.pytorch.serve.grpc.inference.PredictionsRequest)
  })
_sym_db.RegisterMessage(PredictionsRequest)
_sym_db.RegisterMessage(PredictionsRequest.InputEntry)

PredictionResponse = _reflection.GeneratedProtocolMessageType('PredictionResponse', (_message.Message,), {
  'DESCRIPTOR' : _PREDICTIONRESPONSE,
  '__module__' : 'inference_pb2'
  # @@protoc_insertion_point(class_scope:org.pytorch.serve.grpc.inference.PredictionResponse)
  })
_sym_db.RegisterMessage(PredictionResponse)

TorchServeHealthResponse = _reflection.GeneratedProtocolMessageType('TorchServeHealthResponse', (_message.Message,), {
  'DESCRIPTOR' : _TORCHSERVEHEALTHRESPONSE,
  '__module__' : 'inference_pb2'
  # @@protoc_insertion_point(class_scope:org.pytorch.serve.grpc.inference.TorchServeHealthResponse)
  })
_sym_db.RegisterMessage(TorchServeHealthResponse)

_INFERENCEAPISSERVICE = DESCRIPTOR.services_by_name['InferenceAPIsService']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'P\001'
  _PREDICTIONSREQUEST_INPUTENTRY._options = None
  _PREDICTIONSREQUEST_INPUTENTRY._serialized_options = b'8\001'
  _PREDICTIONSREQUEST._serialized_start=83
  _PREDICTIONSREQUEST._serialized_end=272
  _PREDICTIONSREQUEST_INPUTENTRY._serialized_start=228
  _PREDICTIONSREQUEST_INPUTENTRY._serialized_end=272
  _PREDICTIONRESPONSE._serialized_start=274
  _PREDICTIONRESPONSE._serialized_end=314
  _TORCHSERVEHEALTHRESPONSE._serialized_start=316
  _TORCHSERVEHEALTHRESPONSE._serialized_end=358
  _INFERENCEAPISSERVICE._serialized_start=361
  _INFERENCEAPISSERVICE._serialized_end=602
# @@protoc_insertion_point(module_scope)

import grpc

from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2

class InferenceAPIsServiceStub(object):
	"""Missing associated documentation comment in .proto file."""

	def __init__(self, channel):
		"""Constructor.

		Args:
			channel: A grpc.Channel.
		"""
		self.Ping = channel.unary_unary(
			'/org.pytorch.serve.grpc.inference.InferenceAPIsService/Ping',
			request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
			response_deserializer=TorchServeHealthResponse.FromString,
			)
		self.Predictions = channel.unary_unary(
			'/org.pytorch.serve.grpc.inference.InferenceAPIsService/Predictions',
			request_serializer=PredictionsRequest.SerializeToString,
			response_deserializer=PredictionResponse.FromString,
			)

class InferenceAPIsServiceServicer(object):
	"""Missing associated documentation comment in .proto file."""

	def Ping(self, request, context):
		"""Missing associated documentation comment in .proto file."""
		context.set_code(grpc.StatusCode.UNIMPLEMENTED)
		context.set_details('Method not implemented!')
		raise NotImplementedError('Method not implemented!')

	def Predictions(self, request, context):
		"""Predictions entry point to get inference using default model version.
		"""
		context.set_code(grpc.StatusCode.UNIMPLEMENTED)
		context.set_details('Method not implemented!')
		raise NotImplementedError('Method not implemented!')

def add_InferenceAPIsServiceServicer_to_server(servicer, server):
	rpc_method_handlers = {
		'Ping': grpc.unary_unary_rpc_method_handler(
			servicer.Ping,
			request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
			response_serializer=TorchServeHealthResponse.SerializeToString,
		),
		'Predictions': grpc.unary_unary_rpc_method_handler(
			servicer.Predictions,
			request_deserializer=PredictionsRequest.FromString,
			response_serializer=PredictionResponse.SerializeToString,
		),
	}
	generic_handler = grpc.method_handlers_generic_handler(
		'org.pytorch.serve.grpc.inference.InferenceAPIsService', rpc_method_handlers)
	server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.

class InferenceAPIsService(object):
	"""Missing associated documentation comment in .proto file."""

	@staticmethod
	def Ping(request,
		target,
		options=(),
		channel_credentials=None,
		call_credentials=None,
		insecure=False,
		compression=None,
		wait_for_ready=None,
		timeout=None,
		metadata=None):
		return grpc.experimental.unary_unary(request, target, '/org.pytorch.serve.grpc.inference.InferenceAPIsService/Ping',
			google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
			TorchServeHealthResponse.FromString,
			options, channel_credentials,
			insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

	@staticmethod
	def Predictions(request,
		target,
		options=(),
		channel_credentials=None,
		call_credentials=None,
		insecure=False,
		compression=None,
		wait_for_ready=None,
		timeout=None,
		metadata=None):
		return grpc.experimental.unary_unary(request, target, '/org.pytorch.serve.grpc.inference.InferenceAPIsService/Predictions',
			PredictionsRequest.SerializeToString,
			PredictionResponse.FromString,
			options, channel_credentials,
			insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


def start_torchserving(model_name, model_version,port,protocol):
	
	logging.info(f"Starting torch serving on port {port} using {protocol} protocol")
	if protocol.lower() == "restful":
		target_port = 8505
	elif protocol.lower() == "grpc":	
		target_port = 8508

	kube = FalcoServingKube(model_name,
		port=port,
		targetport=target_port,
		template_type="torch",
		ready_message="WORKER_MODEL_LOADED",
		error_message=None)
			
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
	service_address = kube.get_service_address()
	
	if protocol.lower() == "restful":
		raise NotImplementedError
	elif protocol.lower() == "grpc":	
		tfserver = TorchServinggRPC(model_name,model_version,service_address,kube)
		return tfserver
	else:
		logging.error(f"Couldn't start container for {model_name}")
		return None

class TorchServing:
	def __init__(self,
	model_name,
	model_version,
	service_address,
	kube):
		self._name = model_name
		self._version = model_version
		self._kube = kube
		self._service_address = service_address
		logging.info(f"New torch serving container initialized for {model_name} on {service_address}")

	@property
	def service_address(self):
		return self._service_address

	def post(self):
		raise NotImplementedError
	
	async def post_async(self,session,frame):
		raise NotImplementedError
	
	def is_running(self):
		return self._kube.is_running()

class TorchServinggRPC(TorchServing):
	def __init__(self,
	model_name,
	model_version,
	service_address,
	kube):
		TorchServing.__init__(self,model_name,
		model_version,service_address,kube)
	
		logging.info(f"New gRPC torch serving initialized for {model_name} on {service_address}")

	def post(self,frame):
		raise NotImplementedError
			
	async def post_async(self,stub,frame):
		try:
			frame = Image.fromarray(frame)
			output = io.BytesIO()
			frame.save(output, format='JPEG')
			input_data = {'data': output.getvalue()}
			response = await stub.Predictions(
					PredictionsRequest(model_name=self._name,
					input=input_data))
			prediction = response.prediction
			return prediction
		except Exception as e:
			logging.error(e)
			return None