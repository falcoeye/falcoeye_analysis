import os
import cv2
import grpc
import logging
import traceback
import numpy as np

from ..node import Node
from ...artifact import get_model_server

from ...artifact.k8s.torch import InferenceAPIsServiceStub

from tritonclient.utils import InferenceServerException
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput



class Model(Node):
    
    def __init__(self, name,
        model_name,
        version,
        protocol="gRPC",
        initialize=True
        ):
        Node.__init__(self,name)
        self._model_name = model_name
        self._version = version
        self._model_server = None
        self._protocol = protocol
        if initialize:
            self._init_serving_service()
            if not self._serving_ready:
                logging.warning(f"Model server is off or doesn't exists {model_name}")

    def _init_serving_service(self):
        raise NotImplementedError

    def _is_ready(self):
        if self._serving_ready:
            return True
        else:
            # Try now to init
            self._init_serving_service()
            return self._serving_ready

    def get_service_address(self):
        return self._model_server.service_address

    def get_input_size(self):
        return self._input_size

    def run(self,session):
        if not self._is_ready:
            return
             
        logging.info(f'Predicting {self._data.qsize()} frames')
        while self.more():
            item = self.get()
            
            logging.info(f"New frame to post to container {item.framestamp} {item.timestamp} {item.frame.shape}")
            raw_detections =  self._model_server.post(session,item.frame)
            logging.info(f"Prediction received {item.framestamp}")
            self.sink([item,raw_detections])
        
        logging.info(f"{self._name} completed")

    async def run_on_async(self,session,item):
        """
        safe-node: will not cause the workflow to fail
        """
        if not self._is_ready():
            return None
        try:
            logging.info(f"New frame to post to container {item.framestamp} {item.timestamp}")
            raw_detections =  await self._model_server.post_async(session,item.frame)
            logging.info(f"Prediction received {item.framestamp}")
            return [item,raw_detections]
        except Exception as e:
            return None

    def run_on(self,item):
        """
        safe-node: will not cause the workflow to fail
        """
        if not self._is_ready():
            return None
        try:
            logging.info(f"New frame to post to container {item.framestamp} {item.timestamp}")
            raw_detections =  self._model_server.post(item.frame)
            logging.info(f"Prediction received {item.framestamp}")
            return [item,raw_detections]
        except Exception as e:
            return None

class TFModel(Model):
    def __init__(self, name,
        model_name,
        version,
        protocol="gRPC",
        input_name="input_tensor"
        ):
        Model.__init__(self,name,model_name,
            version,protocol,initialize=False)
        self._input_name = input_name
        self._init_serving_service()

    def _init_serving_service(self):
        logging.warning(f"Initializing model server with input name {self._input_name}")
        self._model_server = get_model_server(self._model_name,
        "tf",self._version,self._protocol,True,
        self._input_name)
        if self._model_server is None:
            logging.warning(f"Model server is off or doesn't exists {self._model_name}")
            self._serving_ready = False
        else:
            logging.info(f"Model server is on for {self._model_name}. Connection established with {type(self._model_server)}")
            self._serving_ready = True

class TorchModel(Model):
    GRPC_MAX_RECEIVE_MESSAGE_LENGTH = 4096*4096*3
    GRPC_OPTIONS  = [
                ('grpc.max_send_message_length', GRPC_MAX_RECEIVE_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', GRPC_MAX_RECEIVE_MESSAGE_LENGTH)]
    def __init__(self, name,
        model_name,
        version,
        protocol="gRPC",

        ):
        Model.__init__(self,name,model_name,version,protocol)
        self._stub = None
        self._chennel = None

    def _init_serving_service(self):
        self._model_server = get_model_server(self._model_name,"torch",self._version,self._protocol)
        if self._model_server is None:
            logging.warning(f"Model server is off or doesn't exists {self._model_name}")
            self._serving_ready = False
        else:
            logging.info(f"Model server is on for {self._model_name}. Connection established with {type(self._model_server)}")
            self._serving_ready = True   
    
    def run(self):
        
        if self._protocol == "gRPC":
            try:
                self._initialize_grpc()
                Model.run(self,self._stub)

            except Exception as e:
                logging.error(e)
        else:
            raise NotImplementedError
    
    def _initialize_grpc(self):
        try:
            host = self.get_service_address() 
            channel_security = os.getenv("CHANNEL","insecure")
            logging.info(f"Starting concurrent gRPC looping for {self.name} on {host} with {channel_security} channel")  
            if channel_security == "secure":
                self._channel = grpc.secure_channel(host,
                    grpc.ssl_channel_credentials(),options=TorchModel.GRPC_OPTIONS)
                self._stub = InferenceAPIsServiceStub(self._channel)
                logging.info(f"Starting stub for {self.name}  tasks in secure_channel") 
            else:
                self._channel = grpc.insecure_channel(host, options=TorchModel.GRPC_OPTIONS)
                self._stub = InferenceAPIsServiceStub(self._channel)
                logging.info(f"Starting stub for {self.name} tasks in insecure_channel") 
        except Exception as e:
            logging.error(e)
    
    def __call__(self,data,as_image=True):
        if self._protocol == "gRPC":
            if not self._is_ready():
                return None

            if self._stub is None:
                self._initialize_grpc()
            
            try:
                #logging.info(
                #    f"New data with shape {data.shape} to post to container")
                raw_detections =  self._model_server.post(self._stub,
                    data,
                    as_image)
                return raw_detections
            except Exception as e:
                logging.error(traceback.format_exc())
                return None
        else:
            raise NotImplementedError

class TritonModel(Model):
    GRPC_MAX_RECEIVE_MESSAGE_LENGTH = 4096*4096*3
    GRPC_OPTIONS = [
        ('grpc.max_send_message_length', GRPC_MAX_RECEIVE_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', GRPC_MAX_RECEIVE_MESSAGE_LENGTH)
    ]

    def __init__(self, name,
                 model_name,
                 version,
                 protocol="gRPC",
                 input_name="images",
                 input_shape=[1, 3, 1280, 1280],
                 input_type="FP32",
                 output_names=["num_dets", "det_boxes", "det_scores", "det_classes"]):
        Model.__init__(self, name, model_name, version, protocol, initialize=False)
        self._input_name = input_name
        self._input_shape = input_shape
        self._input_type = input_type
        self._output_names = output_names
        self._init_serving_service()

    def _init_serving_service(self):
        """Initialize connection to Triton model server"""
        self._model_server = get_model_server(
            self._model_name,
            "triton",
            self._version,
            self._protocol,
            run_if_down=True
        )
        if self._model_server is None:
            logging.warning(f"Model server is off or doesn't exist {self._model_name}")
            self._serving_ready = False
        else:
            logging.info(f"Model server is on for {self._model_name}. Connection established with {type(self._model_server)}")
            self._serving_ready = True

    def _prepare_inference_input(self, image_data):
        """Prepare the input data for inference"""
        inputs = []
        outputs = []
        
        # Expect image_data to already be preprocessed
        inputs.append(InferInput(self._input_name, self._input_shape, self._input_type))
        inputs[0].set_data_from_numpy(np.expand_dims(image_data, axis=0))
        
        for output_name in self._output_names:
            outputs.append(InferRequestedOutput(output_name))
            
        return inputs, outputs

    def _process_frame(self, frame_data):
        """Process a single frame"""
        try:
            # Prepare inference input (frame should already be preprocessed)
            inputs, outputs = self._prepare_inference_input(frame_data)
            
            # Run inference using model server
            results = self._model_server.post(client, inputs, outputs)
            if results is None:
                return None
                
            # Convert InferResult to the expected dictionary format
            return {
                'num_dets': results.as_numpy('num_dets'),
                'det_boxes': results.as_numpy('det_boxes'),
                'det_scores': results.as_numpy('det_scores'),
                'det_classes': results.as_numpy('det_classes')
            }
            
        except Exception as e:
            logging.error(f"Unexpected error during inference: {str(e)}")
            traceback.print_exc()
            return None

    def run(self):
        """Main inference loop"""
        if not self._is_ready():
            return
            
        logging.info(f'Processing {self._data.qsize()} frames with Triton model')
        while self.more():
            item = self.get()
            try:
                logging.info(f"New frame to process: {item.framestamp} {item.timestamp}")
                raw_detections = self._process_frame(item.frame, None)  # client will be managed by thread wrapper
                logging.info(f"Prediction received for frame {item.framestamp}")
                self.sink([item, raw_detections])
            except Exception as e:
                logging.error(f"Error processing frame: {str(e)}")
                continue
                
        logging.info(f"{self._name} completed")

    async def run_on_async(self, client, item):
        if not self._is_ready():
            return None
            
        try:
            logging.info(f"Processing frame asynchronously: {item.framestamp} {item.timestamp}")
            input_data = item.tensor.astype(np.float32)
            inputs, outputs = self._prepare_inference_input(input_data)
            
            # Get raw inference result
            raw_result = await self._model_server.post_async(client, inputs, outputs)
            
            if raw_result is None:
                return None
                
            # Convert to dictionary format
            detections = {
                'num_dets': raw_result.as_numpy('num_dets'),
                'det_boxes': raw_result.as_numpy('det_boxes'),
                'det_scores': raw_result.as_numpy('det_scores'),
                'det_classes': raw_result.as_numpy('det_classes')
            }
            
            # Log shapes for debugging
            logging.info(f"Detection shapes: num_dets={detections['num_dets'].shape}, "
                        f"boxes={detections['det_boxes'].shape}, "
                        f"scores={detections['det_scores'].shape}, "
                        f"classes={detections['det_classes'].shape}")
                        
            return [item, detections]
            
        except Exception as e:
            logging.error(f"Error in async processing: {str(e)}")
            return None

    def run_on(self, item):
        if not self._is_ready():
            return None
            
        try:
            logging.info(f"Processing frame: {item.framestamp} {item.timestamp}")
            raw_detections = self._process_frame(item.frame, None)  # client will be managed by thread wrapper
            return [item, raw_detections] if raw_detections is not None else None
        except Exception as e:
            logging.error(f"Error in processing: {str(e)}")
            return None

    def __call__(self, data, as_image=True):
        if not self._is_ready():
            return None
            
        try:
            raw_detections = self._process_frame(data if as_image else data.frame, None)  # client will be managed by thread wrapper
            return raw_detections
        except Exception as e:
            logging.error(traceback.format_exc())
            return None