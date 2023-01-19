from ..node import Node
import logging

import numpy as np


class BoundingBoxDrawer(Node):
    def __init__(self,name,cmap,translate=True):
        Node.__init__(self,name)
        self._cmap = cmap
        self._initialized = False
        self._width = -1
        self._height = -1
        self._translate = translate
    
    def run(self):
        # expecting items of type FalcoeyeDetection
        while self.more():
            item = self.get()
            # TODO: assert different size, and try to remove somehow
        
            
            n = item.count
            logging.info(f"Running {self._name} on item {item.framestamp} with {n} boxex")
            for i in range(n):
                cl = item.get_class(i)
                color = self._cmap[cl]
                item.draw_bounding_box(i,color,translate=self._translate)
            self.sink(item)