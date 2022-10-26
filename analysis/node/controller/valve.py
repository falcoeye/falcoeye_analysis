
from ..node import Node
from threading import Thread
import logging



class OneLeakyOneTimedValve(Node):
    def __init__(self,name,nodes,
        timed_gate_open_freq=30,
        timed_gate_opened_last=10,
        close_on_close=[False,True]):
        Node.__init__(self,name)
        self._nodes = nodes
        self._leaky_gate_node = nodes[0]
        self._timed_gate_node = nodes[1]
        self._timed_gate_open_freq = timed_gate_open_freq
        self._timed_gate_opened_last = timed_gate_opened_last
        self._close_on_close = close_on_close
        self._counter = 0
           
    def run_forever_(self):
        """
        Critical node: failure here should cause the workflow to fail
        """
        logging.info(f"Starting looping for {self.name}")
        try:
            while self._continue or self.more():
                if not self.more():
                    continue
                item = self.get()
                # TODO: eliminate ._data
                logging.info(f"New item entered the valve {self._name}")
                if self._counter < self._timed_gate_opened_last:
                    logging.info(f"Directing {self._counter} to timed gate")
                    self._timed_gate_node._data.put(item)
                    self._counter += 1
                elif self._counter == self._timed_gate_open_freq:
                    logging.info(f"Directing {self._counter} to timed gate and closing")
                    self._counter = 0
                    self._timed_gate_node._data.put(item)
                else:
                    logging.info(f"Directing {self._counter} to leaky gate")
                    self._leaky_gate_node._data.put(item)
                    self._counter += 1
            
            logging.info(f"No more from source. Closing volve {self.name}")     
            if self._done_callback:
                self._done_callback(self._name)

            for c,n in zip(self._close_on_close,self._nodes):
                if c:
                    n.close()
        except Exception as e:
            logging.error(e)
            self._error_callback(self._name,str(e))
        
    def run_async(self,done_callback,error_callback):
        self._done_callback = done_callback
        self._error_callback = error_callback
        self._continue = True
        self._thread = Thread(target=self.run_forever_, args=(),daemon=True)
        self._thread.start()