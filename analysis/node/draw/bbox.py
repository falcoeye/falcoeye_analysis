from ..node import Node
import logging

import numpy as np


class BoundingBoxDrawer(Node):
    def __init__(self,name,cmap):
        Node.__init__(self,name)
        self._cmap = cmap
        self._initialized = False
        self._width = -1
        self._height = -1
    
    def run(self):
        # expecting items of type FalcoeyeDetection
        while self.more():
            item = self.get()
            # TODO: assert different size, and try to remove somehow
        
            logging.info(f"Running {self._name} on item {item.framestamp}")
            n = item.count
            for i in range(n):
                cl = item.get_class(i)
                color = self._cmap[cl]
                item.draw_bounding_box(i,color)
            self.sink(item)