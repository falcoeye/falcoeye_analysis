from queue import Queue
import logging
class Node:
    def __init__(self,name):
        self._name = name
        self._sinks = []
        self._data = Queue()
        self._thread = None
        self._done_callback = None
        self._continue = False
    
    def close_sinks(self):
        for sink in self._sinks:
            sink.close()

    @property
    def name(self):
        return self._name

    def add_sink(self,sink):
        self._sinks.append(sink)
    
    def run(self):
        raise NotImplementedError
    
    def run_async(self,done_callback,error_callback):
        raise NotImplementedError
    
    def sink(self,data):
        #logging.info(f"Sinking from {self._name} to {len(self._sinks)} sinks")
        for sink in self._sinks:
            #logging.info(f"Sinking from {self._name} to {sink._name}")
            sink._data.put(data)
    
    def open(self):
        logging.info(f"Opening {self.name}")
        self._continue = True
        
    def close(self):
        logging.info(f"Closing {self.name}")
        self._continue = False
    
    def more(self):
        return self._data.qsize() > 0
    
    def get(self):
        return self._data.get()