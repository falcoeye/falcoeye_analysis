
from ..node import Node
import logging
class TypeFilter(Node):
    def __init__(self, name, keys):
        Node.__init__(self,name)
        self._keys = keys

    def run(self):
        # expecting items of type FalcoeyeDetction
        while self.more():
            item = self.get()
            # TODO: maybe optimize in one loop 
            classes = item.get_classes()
            for i,c in enumerate(classes):
                if c not in self._keys:
                    item.delete(i)

class SizeFilter(Node):
    def __init__(self, name, width_threshold,height_threshold):
        Node.__init__(self,name)
        self._width_threshold = width_threshold
        self._height_threshold = height_threshold

    def run(self):
        # expecting items of type FalcoeyeDetction
        logging.info(f"Running {self.name} with {self._width_threshold} and {self._height_threshold}")
        while self.more():
            item = self.get()   
            index = 0
            logging.info(f"Before size filter {item.count}")
            for _ in range(item.count):
                if item.iwidth(index) > self._width_threshold or item.iheight(index) > self._height_threshold:
                    item.delete(index)
                else:
                    # when deleting, no index increament due to shift in array
                    index += 1
            self.sink(item)
            logging.info(f"After size filter {item.count}")
