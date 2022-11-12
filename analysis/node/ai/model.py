from ..node import Node
from ...artifact import get_model_server
import logging
import grpc
from ...artifact.k8s.torch import InferenceAPIsServiceStub
import os

class Model(Node):
    
    def __init__(self, name,
        model_name,
        version,
        protocol="gRPC"
        ):
        Node.__init__(self,name)
        self._model_name = model_name
        self._version = version
        self._model_server = None
        self._protocol = protocol
        
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
        protocol="gRPC"
        ):
        Model.__init__(self,name,model_name,version,protocol)

    def _init_serving_service(self):
        
        self._model_server = get_model_server(self._model_name,"tf",self._version,self._protocol)
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
        protocol="gRPC"
        ):
        Model.__init__(self,name,model_name,version,protocol)

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
                host = self.get_service_address() 
                channel_security = os.getenv("CHANNEL","insecure")
                logging.info(f"Starting concurrent gRPC looping for {self.name} on {host} with {channel_security} channel")  
                if channel_security == "secure":
                    channel = grpc.secure_channel(host,
                        grpc.ssl_channel_credentials(),options=TorchModel.GRPC_OPTIONS)
                    stub = InferenceAPIsServiceStub(channel)
                    logging.info(f"Starting stub for {self.name}  tasks in secure_channel") 
                else:
                    channel = grpc.insecure_channel(host, options=TorchModel.GRPC_OPTIONS)
                    stub = InferenceAPIsServiceStub(channel)
                    logging.info(f"Starting stub for {self.name} tasks in insecure_channel") 

                Model.run(self,stub)

            except Exception as e:
                logging.error(e)
        else:
            raise NotImplementedError