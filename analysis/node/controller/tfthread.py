from tensorflow_serving.apis import prediction_service_pb2_grpc
from .thread import ConcurrentRequestTaskThreadWrapper
import asyncio
from grpc import aio
import grpc
import logging
import aiohttp
import os


class ConcurrentTFgRPCTasksThreadWrapper(ConcurrentRequestTaskThreadWrapper):
    GRPC_MAX_RECEIVE_MESSAGE_LENGTH = 4096*4096*3
    def __init__(self,name,node,ntasks=2,max_send_message_length=6220800):
        ConcurrentRequestTaskThreadWrapper.__init__(self,name,node,ntasks)
        self._max_send_message_length = max_send_message_length
        self._options  = [
                    ('grpc.max_send_message_length', ConcurrentTFgRPCTasksThreadWrapper.GRPC_MAX_RECEIVE_MESSAGE_LENGTH),
                    ('grpc.max_receive_message_length', ConcurrentTFgRPCTasksThreadWrapper.GRPC_MAX_RECEIVE_MESSAGE_LENGTH)]
    async def run_forever_(self):
        """
        Critical node: failure here should cause the workflow to fail
        """


        try:
            host = self._node.get_service_address()
            
            logging.info(f"Starting concurrent gRPC looping for {self.name} on {host}")  
            if os.getenv("CHANNEL","insecure") == "secure":
                async with aio.secure_channel(host, 
                    grpc.ssl_channel_credentials(), options=self._options) as channel:
                    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
                    logging.info(f"Starting stub looping for {self.name} with {self._ntasks} tasks in secure_channel") 
                    await self.run_session_loop_(stub)
            else:
                async with aio.insecure_channel(host, options=self._options) as channel:
                    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
                    logging.info(f"Starting stub looping for {self.name} with {self._ntasks} tasks in insecure_channel") 
                    await self.run_session_loop_(stub)
            self._loop.stop()
            
            logging.info(f"Loop {self.name} inturrepted. Flushing queue")
            if self._done_callback:
                self._done_callback(self._name)  
            self.close_sinks() 
        except Exception as e:
            logging.error(e)
            self._error_callback(self._name,str(e))
