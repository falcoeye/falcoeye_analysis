from ..node import Node
from PIL import ImageDraw, Image
import numpy as np
import logging
import pandas as pd

class ZoneCounter(Node):
    def __init__(self, name,points,keys):
        Node.__init__(self,name)

        if type(points) == str:
            points = points.split(",")        
            points = [[int(points[i]),int(points[i+1])] for i in range(0,len(points),2)]
        
        logging.info(f"Creating zone counter around {points} with keys {keys}")
        self._points = points
        self._mask = None
        self._keys = keys
        # +2 for Timestamp,Frame_Order
        self._len = len(keys)+2
        self._keys_index = {k:i+2 for i,k in enumerate(self._keys)}
        self._initilized = False
    
    def initialize(self,width,height):
        self._mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(self._mask).polygon(
            [tuple(x) for x in self._points], outline=1, fill=1
        )
        self._mask = np.array(self._mask).astype(bool)
        self._width, self._height = width, height
        self._initilized = True

    def translate_pixel(self, x, y):
        return int(x * self._width), int(y * self._height)

    def run(self):
        # expecting items of type FalcoeyeDetction
        logging.info(f"Running {self.name}")
        c = lambda x, y: self._mask[y, x]
        table = []
        while self.more():
            item = self.get()
            
            # TODO: assert different size, and try to remove somehow
            if not self._initilized:
                # assuming object with Falcoeye wrapper
                height, width,_ = item.size
                self.initialize(width,height)
            
            row = np.zeros(self._len)
            row[:2] = [item.timestamp,item.framestamp]
            n = item.count
            for i in range(n):
                xmin, ymin, xmax, ymax = item.get_box(i)
                xmin, ymin = self.translate_pixel(xmin * 0.9999, ymin * 0.9999)
                xmax, ymax = self.translate_pixel(xmax * 0.9999, ymax * 0.9999)
                if (c(xmin, ymin) or c(xmin, ymax) or c(xmax, ymin) or c(xmax, ymax)):
                    cl = item.get_class(i)
                    if cl in self._keys:
                        row[self._keys_index[cl]] += 1
            table.append(row)

        df = pd.DataFrame(table,columns=["Timestamp","Frame_Order"]+self._keys)
        logging.info(f"\n{df}")
        self.sink(df)

    def run_once(self,item):
        # expecting items of type FalcoeyeDetction
        logging.info(f"Running {self.name}")
        # TODO: move this for god sake
        c = lambda x, y: self._mask[y, x]    
        # TODO: assert different size, and try to remove somehow
        if not self._initilized:
            # assuming object with Falcoeye wrapper
            height, width,_ = item.size
            logging.info(f"Initializing {self._name} with {width}X{height}")
            self.initialize(width,height)

        row = np.zeros(self._len)
        row[:2] = [item.timestamp,item.framestamp]
        n = item.count
        for i in range(n):
            xmin, ymin, xmax, ymax = item.get_box(i)
            xmin, ymin = self.translate_pixel(xmin * 0.9999, ymin * 0.9999)
            xmax, ymax = self.translate_pixel(xmax * 0.9999, ymax * 0.9999)
            if (c(xmin, ymin) or c(xmin, ymax) or c(xmax, ymin) or c(xmax, ymax)):
                cl = item.get_class(i)
                if cl in self._keys:
                    row[self._keys_index[cl]] += 1
        return row

class ZonesCounter(Node):
    def __init__(self, name,zones,keys):
        Node.__init__(self,name)
        self._zones = []
        self._keys = keys
        if type(zones) == str:
            zones = zones.split(";")
            for zs in zones:
                name,points = zs.split(":")
                points = points.split(",")
                points = [[int(points[i]),int(points[i+1])] for i in range(0,len(points),2)]
                self._zones.append(ZoneCounter(name,points,keys))

    def run(self):
        # expecting items of type FalcoeyeDetction
        logging.info(f"Running {self.name}")
        data = []
        zones = []
        while self.more():
            item = self.get()
            for z in self._zones:
                data.append(z.run_once(item))
                zones.append(z.name)
        if len(data) > 0:
            df = pd.DataFrame()
            df["Zone"] = zones
            df[["Timestamp","Frame_Order"]+self._keys] = data
            self.sink(df)