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
    def __init__(self,frame,heatmap,pafs,resize_factor=8):
        FalcoeyeAIWrapper.__init__(self,frame)
        self._boxes = None
        self._skeletons = None
        self._keypoints = None
        self._heatmap = heatmap
        self._pafs = pafs
        self._resize_factor = resize_factor
        logging.info(f"Calculating skeleton for frame {frame.framestamp}")
        self._coordinates = get_coordinates(BODY_CONFIG, self._heatmap)
        self._connections = get_connections(BODY_CONFIG, self._coordinates, self._pafs)
        try:
            skeletons = estimate(BODY_CONFIG, self._connections)
            self._skeletons = []
            self._boxes = []
            self._keypoints = []
            xy_by_id = dict([(item[3], np.array([item[0]/self._heatmap.shape[1], item[1]/self._heatmap.shape[0]])) for sublist in self._coordinates.values() for item in sublist])
            for i,s in enumerate(skeletons):
                s = s.astype(np.int32)
                keypoints = np.zeros((18,3))
                for j,k in enumerate(s[:-2]):
                    if k != -1:
                        keypoints[j] = (j,xy_by_id[k][0],xy_by_id[k][1]) 
                    else:
                        keypoints[j] = (j,0,0)
                
                # to handle empty skeleton
                # not sure why estimate return empty skeleton somtime
                if keypoints[:,1].sum() == 0 or keypoints[:,2].sum()==0:
                    continue

                b = self._get_bbox_from_keypoints(keypoints)
                if not b:
                    continue
                self._skeletons.append(s)
                self._boxes.append(b)
                self._keypoints.append(keypoints)
                
            # for tracking
            logging.info(f"Skeleton for frame {frame.framestamp} calculated")
        except Exception as e:
            logging.info(f"Error in calculating skeleton for frame {frame.framestamp}")
            logging.exception(e)

    def _get_bbox_from_keypoints(self,keypoints):
        def expand_bbox(xmin, xmax, ymin, ymax,img_h,img_w ):
            """expand bbox for containing more background"""
            width = xmax - xmin
            height = ymax - ymin
            ratio = 0.1   # expand ratio
            new_xmin = np.clip(xmin - ratio * width, 0, img_w)
            new_xmax = np.clip(xmax + ratio * width, 0, img_w)
            new_ymin = np.clip(ymin - ratio * height, 0, img_h)
            new_ymax = np.clip(ymax + ratio * height, 0, img_h)
            new_width = new_xmax - new_xmin
            new_height = new_ymax - new_ymin
            return [new_xmin, new_ymin, new_width, new_height]

        img_h,img_w = self.size[:2]
        
        keypoints = np.where(keypoints[:, 1:] !=0, keypoints[:, 1:], np.nan)
        keypoints[:, 0] *= img_w
        keypoints[:, 1] *= img_h
        xmin = np.nanmin(keypoints[:,0])
        ymin = np.nanmin(keypoints[:,1])
        xmax = np.nanmax(keypoints[:,0])
        ymax = np.nanmax(keypoints[:,1])
        
        bbox = expand_bbox(xmin, xmax, ymin, ymax,img_h,img_w)
        # discard bbox with width and height == 0
        if bbox[2] < 1 or bbox[3] < 1 :
            return None
        return bbox

    def get_boxes(self):
        return self._boxes

    def get_box(self,index):
        # TODO: check if boxes is defined
        return self._boxes[index]
    
    def set_box(self,index,box):
        self._boxes[index] = box

    def get_flatten_keypoints(self,index):
        flatten_keypoints = self._keypoints[index][:,1:].flatten()
        return flatten_keypoints

    def draw(self):
        # since open pose has 3 max pooling layers, resize_fac = 8
        # size = (n-filter_size)/stride_size
        if self._skeletons is not None:
            self._frame.set_frame(draw(BODY_CONFIG, self._frame.frame,
                self._coordinates, self._skeletons, resize_fac=8))
    
    def delete(self,index):
        if type(index) == int:
            del self._skeletons[index]
            del self._boxes[index]
        elif type(index) == list:
            for i in sorted(index, reverse=True):
                del self._skeletons[i]
                del self._boxes[i]

    def __lt__(self,other):
        if type(other) == FalcoeyeOpenPoseHPE:
            return self._frame < other._frame     
        return FalcoeyeAIWrapper.__lt__(self,other)
    
    def __eq__(self,other):
        if type(other) == FalcoeyeOpenPoseHPE:
            return self._frame == other._frame
        return FalcoeyeAIWrapper.__eq__(self,other)

    def __len__(self):
        n = 0
        if self._skeletons is not None:
            n = len(self._skeletons)
        return n

    def __get__(self,index):
        return self._skeletons[index]
    
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
                
                ophpe = FalcoeyeOpenPoseHPE(frame,heatmap,paf,self._resize_factor)
                self.sink(ophpe)
            except Exception as e:
                logging.exception(e)
