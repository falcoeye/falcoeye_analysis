from ..node import Node
import logging
from PIL import ImageDraw, Image
import numpy as np

class StaticPolygonDrawer(Node):
    def __init__(self,name,points,color,alpha=0.5):
        Node.__init__(self,name)
        self._alpha = alpha
        self._points = points
        self._color = color
    
    def run_once(self,frame):
        # assuming FalcoeyeFrame
        pass

class StaticPolygonsDrawer(Node):
    def __init__(self, name,polygons,color=(0,0,255),alpha=0.5):
        Node.__init__(self,name)
        self._initialized = False
        self._mask = None
        self._width = -1
        self._height = -1
        self._color = color
        self._alpha = alpha
        self._polygons = []
        if type(polygons) == str:
            polygons = polygons.split(";")
            for p in polygons:
                name,points = p.split(":")
                points = points.split(",")
                points = [[int(points[i]),int(points[i+1])] for i in range(0,len(points),2)]
                self._polygons.append((name,points))
        else:
            raise NotImplementedError

    def initialize(self,width,height):
        self._mask = Image.new("RGB", (width, height), 0)
        for name,points in self._polygons:
            ImageDraw.Draw(self._mask).polygon(
                [tuple(x) for x in points], outline=1, fill=self._color
            )
        self._mask = np.asarray(self._mask)
        self._width, self._height = width, height
        self._initilized = True

    def run(self):
        # expecting items of type FalcoeyeFrame or a wrapper for it
        while self.more():
            item = self.get()
            # TODO: assert different size, and try to remove somehow
            if not self._initialized:
                height, width,_ = item.size
                self.initialize(width,height)
            # assuming FalcoeyeFrame or a wrapper for it
            logging.info(f"Running {self._name} on item {item.framestamp}")
            item.blend(self._mask,alpha=self._alpha,inplace=True)
           
            self.sink(item)

