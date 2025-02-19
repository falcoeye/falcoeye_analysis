from .thread import ConcurrentRequestTaskThreadWrapper
import asyncio
from tritonclient.grpc import aio as triton_aio
import grpc
import logging
import os

class ConcurrentTritongRPCTasksThreadWrapper(ConcurrentRequestTaskThreadWrapper):
    def __init__(self, name, node, ntasks=2, max_send_message_length=20*1024*1024):  # 20MB default
        ConcurrentRequestTaskThreadWrapper.__init__(self, name, node, ntasks)
        self._max_message_length = max_send_message_length
        self._options = [
            ('grpc.max_send_message_length', self._max_message_length),
            ('grpc.max_receive_message_length', self._max_message_length)
        ]
        
    async def run_forever_(self):
        """
        Critical node: failure here should cause the workflow to fail
        """
        try:
            host = self._node.get_service_address()
            channel_security = os.getenv("CHANNEL", "insecure")
            logging.info(f"Starting concurrent Triton gRPC looping for {self.name} on {host} with {channel_security} channel")
            
            # Create Triton client with appropriate channel security
            if channel_security == "secure":
                client = triton_aio.InferenceServerClient(
                    url=host,
                    ssl=True,
                    verbose=False,
                    channel_args=self._options
                )
            else:
                client = triton_aio.InferenceServerClient(
                    url=host,
                    ssl=False,
                    verbose=False,
                    channel_args=self._options
                )
                
            # Check server health
            if not await client.is_server_live():
                raise Exception("Triton server is not live")
            if not await client.is_server_ready():
                raise Exception("Triton server is not ready")
            if not await client.is_model_ready(self._node._model_name):
                raise Exception(f"Model {self._node._model_name} is not ready")
                
            logging.info(f"Starting Triton client loop for {self.name} with {self._ntasks} tasks")
            await self.run_session_loop_(client)
            
            self._loop.stop()
            logging.info(f"Loop {self.name} interrupted. Flushing queue")
            
            if self._done_callback:
                self._done_callback(self._name)
            self.close_sinks()
            
        except Exception as e:
            logging.error(f"Error in Triton thread wrapper: {str(e)}")
            self._error_callback(self._name, str(e))
            
    async def process_item_(self, client, item):
        """
        Process a single item using the Triton client
        """
        try:
            result = await self._node.run_on_async(client, item)
            if result:
                self.sink(result)
        except Exception as e:
            logging.error(f"Error processing item in Triton thread: {str(e)}")