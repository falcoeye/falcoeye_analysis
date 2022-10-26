from ..node import Node
from threading import Thread
import logging
import aiohttp

import asyncio
from grpc import aio
import grpc


class ThreadWrapper(Node):
    def __init__(self,name,node):
        Node.__init__(self,name)
        self._node = node

    def run_forever_(self):
        """
        Critical node: failure here should cause the workflow to fail
        """
        try:
            logging.info(f"Starting looping for {self.name}")
            while self._continue or self.more():
                if not self.more():
                    continue
                logging.info(f"New item for {self.name} sequence")
                item = self.get()
                node_res = self._node.run_on(item)
                self.sink(node_res)
            
            logging.info(f"Loop {self.name} inturrepted. Flushing queue")
            if self._done_callback:
                self._done_callback(self._name)  
            self.close_sinks() 
        except Exception as e:
            logging.error(e)
            self._error_callback(self._name,str(e))

    def run_async(self,done_callback,error_callback):
        self._done_callback = done_callback
        self._error_callback = error_callback
        self._continue = True
        self._thread = Thread(target=self.run_forever_, args=(),daemon=True)
        self._thread.start()

class ConcurrentRequestTaskThreadWrapper(Node):
    def __init__(self,name,node,ntasks=2):
        Node.__init__(self,name)
        self._node = node
        self._loop = None
        self._ntasks = ntasks
    
    async def task_(self,session,item):
        logging.info(f"Running task asyncronously for frame {item.framestamp}")
        o = await self._node.run_on_async(session,item)
        self.sink(o)
    
    def start_background_loop(self,loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def run_async(self,done_callback,error_callback):
        logging.info(f"Running {self.name} asyncronously")
        self._done_callback = done_callback
        self._error_callback = error_callback
        self._continue = True
        self._loop = asyncio.new_event_loop()     
        self._thread = Thread(target=self.start_background_loop,
                args=(self._loop,),
                daemon=True)
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self.run_forever_(), self._loop)
 
    def run_forever_(self):
        logging.error("Not implemented")

    async def run_session_loop_(self,session):
        """
        Critical node: failure here should cause the workflow to fail
        """
        try:
            tasks = []
            while self._continue or self.more():
                if self.more():
                    logging.info("New data to sink")
                    item = self.get()
                    logging.info(f"Creating task for frame {item.framestamp} with size {item.size}")
                    task = asyncio.create_task(self.task_(session,item))
                    tasks.append(task)
                    logging.info(f"Task created for frame {item.framestamp}")
                if len(tasks) == self._ntasks:
                    logging.info(f"Gathering {self._ntasks} new tasks")
                    await asyncio.gather(*tasks)
                    tasks = []
            if len(tasks) > 0:
                logging.info("Gathering remaining tasks")
                await asyncio.gather(*tasks)
                tasks = []
            logging.info("Exiting sinking loop")
        except Exception as e:
            logging.error(e)
            self._error_callback(self._name,str(e))

class ConcurrentPostTasksThreadWrapper(ConcurrentRequestTaskThreadWrapper):
    
    def __init__(self,name,node,ntasks=2):
        ConcurrentRequestTaskThreadWrapper.__init__(self,name,node,ntasks)
        self._tcplimit = ntasks

    async def run_forever_(self):
        logging.info(f"Starting concurrent looping for {self.name}")        
        
        connector = aiohttp.TCPConnector(limit=self._tcplimit)
        async with aiohttp.ClientSession(connector=connector) as session:
            logging.info(f"Starting aiohttp looping for {self.name} with {self._ntasks} tasks") 
            await self.run_session_loop_(session)

        self._loop.stop()
        logging.info(f"Loop {self.name} inturrepted. Flushing queue")
        if self._done_callback:
            self._done_callback(self._name)
        self.close_sinks() 

