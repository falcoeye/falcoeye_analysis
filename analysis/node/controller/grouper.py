from ..node import Node
from threading import Thread
import logging
import queue
import traceback


class SequenceRunner(Node):
    def __init__(self,name,nodes,frequency=1):
        Node.__init__(self,name)
        self._nodes = nodes
        self._frequency = int(frequency)
        self._counter = 0
        self._sinks = [self._nodes[0]]
    
    def run_(self):
        self._counter += 1
        #logging.info(f"Buffering sequence {self._counter}/{self._frequency}")
        if self._counter == self._frequency: 
            #logging.info(f"Runing sequence now")
            for n in self._nodes:
                #logging.info(f"Runing node {n.name}")
                n.run()  
            self._counter = 0

    def run(self):

        self.open_nodes()
        while self.more():
            item = self.get()
            self.sink(item)
            self.run_()
        if self._counter != 0:
            # tricking the run_ function to force it to run
            self._counter = self._frequency - 1
            self.run_()
        self.close_nodes()
        # run last time to flush. Some nodes have post close scenarios
        logging.info("Flushing nodes after closing.")
        for n in self._nodes:
            logging.info(f"Runing node {n.name}")
            n.run()  
          
    def run_forever_(self):
        """
        Critical node: failure here should cause the workflow to fail
        """
        try:
            logging.info(f"Running forever for {self.name}")

            # opening the nodes (i.e. setting continue to True)
            logging.info(f"Opening sequence nodes for {self.name}")
            self.open_nodes()
            logging.info(f"Starting looping for {self.name}")
            while self._continue or self.more():
                if not self.more():
                    continue

                #logging.info(f"New item for {self.name} sequence")
                item = self.get()
                self.sink(item)
                self.run_()
            logging.info(f"Sequence {self.name} inturrepted.")
            # flushing the remaining
            if self._counter != 0:
                logging.info(f"Flushing queue for {self.name}. {self._counter} item not processed")
                # tricking the run_ function to force it to run
                self._counter = self._frequency - 1
                self.run_()
            logging.info(f"Sequence {self.name} loop exited")
            self._counter = 0

            # TODO: refactor this
            # run last time to flush. Some nodes have post close scenarios
            logging.info("Flushing nodes after closing.")
            for n in self._nodes:
                logging.info(f"Runing node {n.name}")
                n.run() 
            
            # closing the nodes (i.e. setting continue to False)
            self.close_nodes()
            
            if self._done_callback:
                self._done_callback(self._name)

        except Exception as e:
            logging.error(e)
            logging.error(traceback.format_exc())
            self._error_callback(self._name,str(e))
    
    def open_nodes(self):
        for node in self._nodes:
            node.open() 

    def close_nodes(self):
        for node in self._nodes:
            node.close() 

    def run_async(self,done_callback,error_callback):
        # TODO: do we need try/except here?!
        logging.info(f"Running {self.name} async")
        self._done_callback = done_callback
        self._error_callback = error_callback
        self._continue = True
        self._thread = Thread(target=self.run_forever_,daemon=True)
        self._thread.start()
        logging.info(f"Thread for {self.name} started")

class SortedSequence(Node):
    def __init__(self,name):
        Node.__init__(self,name)
        self._data = queue.PriorityQueue()
        self._expected_start = 0
    
    def run(self):
        """
        Safe node: input is assumed to be valid
        """
        logging.info(f"Running SortedSequence {self._name}")
        while self.more():
            if self._data.queue[0] == self._expected_start:
                logging.info(f"Expected root {self._expected_start} so sink. Size {self._data.qsize()}")
                self._expected_start += 1
                item = self.get()
                self.sink(item)
            else:
                logging.info(f"Expected root {self._expected_start} not yet available. Size {self._data.qsize()}")
                break
            
    
    def open(self):
        self._expected_start = 0

    def close(self):
        self._expected_start = 0
        Node.close(self)


