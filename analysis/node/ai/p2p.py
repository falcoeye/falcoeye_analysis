from ..node import Node
from ...artifact import get_model_server
import logging
import numpy as np
from PIL import Image
from .wrapper import FalcoeyeAIWrapper

class FalcoeyeP2P(FalcoeyeAIWrapper):
    
    def __init__(self,frame,points):
        FalcoeyeAIWrapper.__init__(self,frame)
        self._points = points

    @property
    def count(self):
        return self._points.shape[0]

    def __lt__(self,other):
        if type(other) == FalcoeyeP2P:
            return self._frame < other._frame     
        return FalcoeyeAIWrapper.__lt__(self,other)
    
    def __eq__(self,other):
        if type(other) == FalcoeyeP2P:
            return self._frame == other._frame
        return FalcoeyeAIWrapper.__eq__(self,other)

class FalcoeyeP2PNode(Node):
    
    def __init__(self, name,min_score_thresh,max_points):
        Node.__init__(self,name)
        self._min_score_thresh = min_score_thresh
        self._max_points = max_points
        
    def translate(self,detections):

        if type(detections) == bytes:
            detections = np.frombuffer(detections,dtype=np.float32).reshape(-1,3)
        elif type(detections) == list:
            detections = np.array(detections)
        else:
            # Acting safe
            return np.array([])

        points = detections[:,:2]
        scores = detections[:,2]
        points = points[scores > self._min_score_thresh]
        return points

    def run(self):
        """
        Safe node: the input is assumed to be valid or can be handled properly, no need to catch
        """
        logging.info(f"Running falcoeye p2p")
        while self.more():
            item = self.get()
            frame,raw_detections = item
            logging.info(f"New frame for falcoeye p2p  {frame.framestamp} {frame.timestamp}")
            points = self.translate(raw_detections)
            fe_p2p = FalcoeyeP2P(frame,points)
            self.sink(fe_p2p)
