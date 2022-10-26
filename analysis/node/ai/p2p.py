from ..node import Node
from ...artifact import get_model_server
import logging
import numpy as np
from PIL import Image


class FalcoeyeP2P:
    def __init__(self,frame,points):
        self._frame = frame
        self._points = points
        
    @property
    def size(self):
        return self._frame.size
    
    @property
    def frame(self):
        return self._frame.frame

    @property
    def frame_bgr(self):
        return self._frame.frame_bgr
    
    @property
    def count(self):
        return self._points.shape[0]

    def save_frame(self,path):
        Image.fromarray(self._frame).save(f"{path}/{self._frame_number}.png")
    
    @property
    def framestamp(self):
        return self._frame.framestamp
    
    @property
    def timestamp(self):
        return self._frame.timestamp

    def __lt__(self,other):
        if type(other) == FalcoeyeDetection:
            return self._frame < other._frame
        else:
            # assuming other is FalcoeyeFrame
            return self._frame < other
    
    def __eq__(self,other):
        if type(other) == FalcoeyeDetection:
            return self._frame == other._frame
        else:
            # assuming other is FalcoeyeFrame or int
            return self._frame == other

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
