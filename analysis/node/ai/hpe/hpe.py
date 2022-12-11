from ...node import Node
import json
import logging
import numpy as np
from ..wrapper import FalcoeyeAIWrapper
from .config import get_default_configuration
from .coordinates import get_coordinates
from .connections import get_connections
from .estimators import estimate
from .renderers import draw

BODY_CONFIG = get_default_configuration()

class FalcoeyeOpenPoseHPE(FalcoeyeAIWrapper):
    def __init__(self,frame,heatmap,pafs):
        FalcoeyeAIWrapper.__init__(self,frame)
        self._heatmap = heatmap
        self._pafs = pafs
        logging.info(f"Calculating skeleton for frame {frame.framestamp}")
        self._coordinates = get_coordinates(BODY_CONFIG, self._heatmap)
        self._connections = get_connections(BODY_CONFIG, self._coordinates, self._pafs)
        try:
            self._skeletons = estimate(BODY_CONFIG, self._connections)
            logging.info(f"Skeleton for frame {frame.framestamp} calculated")
        except Exception as e:
            logging.info(f"Error in calculating skeleton for frame {frame.framestamp}")
            self._skeletons = None
            logging.exception(e)
    
    def draw(self):
        # since open pose has 3 max pooling layers, resize_fac = 8
        # size = (n-filter_size)/stride_size
        if self._skeletons is not None:
            self._frame.set_frame(draw(BODY_CONFIG, self._frame.frame,
                self._coordinates, self._skeletons, resize_fac=8))
    
    def __lt__(self,other):
        if type(other) == FalcoeyeOpenPoseHPE:
            return self._frame < other._frame     
        return FalcoeyeAIWrapper.__lt__(self,other)
    
    def __eq__(self,other):
        if type(other) == FalcoeyeOpenPoseHPE:
            return self._frame == other._frame
        return FalcoeyeAIWrapper.__eq__(self,other)

class FalcoeyeHPENode(Node):
    def __init__(self, name):
        Node.__init__(self,name)
    
    def run(self):
        raise NotImplementedError

class FalcoeyeTFOpenPoseNode(FalcoeyeHPENode):
    def __init__(self, name,resize_factor=8):
        FalcoeyeHPENode.__init__(self,name)
        self._resize_factor = resize_factor
        self._njoints = 19
        self._nlimbs = 38

    def run(self):
        """
        Safe node: the input is assumed to be valid or can be handled properly, no need to catch
        """
        logging.info(f"Running TF OpenPose Node")
        while self.more():
            item = self.get()

            frame,raw_estimation = item
            try:
                if raw_estimation is None:
                    # TODO: should do failover
                    # since open pose has 3 max pooling layers, resize_fac = 8
                    logging.warning(f"No pose estimation for frame {frame.framestamp}")
                    heatmap = np.zeros((frame.size[0]//8,frame.size[1]//8,self._njoints))
                    paf = np.zeros((frame.size[0]//8,frame.size[1]//8,self._nlimbs))
                else:
                    # gRPC response
                    logging.info("gRPC detection")
                    # [1:] for skipping batch axis, we assume 1 input
                    paf_shape = raw_estimation.outputs['output_11'].tensor_shape.dim
                    htm_shape = raw_estimation.outputs['output_12'].tensor_shape.dim
                    #logging.info(f"frame {frame.framestamp} paf shape {paf_shape} htm shape {htm_shape}")
                    paf_shape = tuple([i.size for i in list(paf_shape)][1:])
                    htm_shape = tuple([i.size for i in list(htm_shape)][1:])
                    #logging.info(f"frame {frame.framestamp} paf shape {paf_shape} htm shape {htm_shape}")
                    paf = np.array(
                        raw_estimation.outputs['output_11'].float_val,
                        dtype=np.float32).reshape(paf_shape)
                    heatmap = np.array(
                        raw_estimation.outputs['output_12'].float_val,
                        dtype=np.float32).reshape(htm_shape)
                
                ophpe = FalcoeyeOpenPoseHPE(frame,heatmap,paf)
                self.sink(ophpe)
            except Exception as e:
                logging.exception(e)
