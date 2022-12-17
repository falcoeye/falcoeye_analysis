
from ..node import Node
import logging

class HPEDrawer(Node):
    def __init__(self,name):
        Node.__init__(self,name)
    
    def run(self):
        # expecting items of type FalcoeyeFrame or a wrapper for it
        while self.more():
            item = self.get()
            logging.info(item)
            # assuming FalcoeyeOpenPoseHPE or a wrapper for it
            logging.info(f"Running {self._name} on item {item.framestamp}")
            item.draw()
            self.sink(item)
