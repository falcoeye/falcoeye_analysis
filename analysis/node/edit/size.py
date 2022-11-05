


from ..node import Node
import logging


class Resizer(Node):
    def __init__(self, name,size):
        Node.__init__(self,name)
        self._enabled = True
        if size == "-1":
            self._enabled = False
        sp = size.split(",")
        self._width = int(sp[0])
        self._height = int(sp[1])

    def run(self):
        # expecting items of type FalcoeyeDetction
        #logging.info(f"Running {self._name} with resolution {self._width}X{self._height}")
        while self.more():
            item = self.get()
            if self._enabled:
                # assuming FalcoeyeFrame
                #logging.info(f"Resizing frame to {self._width}X{self._height}")
                item.resize(self._width,self._height)
                logging.info(f"Frame resized {item.size[1]}X{item.size[0]}")
            self.sink(item)
    
    def close(self):
        self.close_sinks()

    




                