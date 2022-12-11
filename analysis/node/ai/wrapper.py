from ..node import Node
import json
import logging
import numpy as np

class FalcoeyeAIWrapper:
    def __init__(self,frame):
        self._frame = frame
    
    @property
    def size(self):
        return self._frame.frame.shape
    
    @property
    def frame(self):
        return self._frame.frame

    @property
    def frame_bgr(self):
        return self._frame.frame_bgr
    
    @property
    def framestamp(self):
        return self._frame.framestamp
        
    @property
    def timestamp(self):
        return self._frame.timestamp

    
    def set_frame(self,frame):
        self._frame.set_frame(frame)
    
    def save_frame(self,path):
        Image.fromarray(self._frame).save(f"{path}/{self._frame_number}.png")

    def __lt__(self,other):
        # assuming other is FalcoeyeFrame or int
        return self._frame < other
    
    def __eq__(self,other):
        return self._frame == other


