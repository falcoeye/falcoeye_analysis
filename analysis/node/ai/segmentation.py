from ..node import Node
import json
import logging
import numpy as np
from PIL import Image
import base64

class FalcoeyeSegmentation:
    def __init__(self,frame,segments,
        obj_ids,obj_names,colors,
        rgb_map,ignore_value):
        self._frame = frame
        self._segments = segments
        self._obj_ids = obj_ids
        self._obj_names = obj_names
        self._colors = colors
        self._rgb_map = rgb_map.astype(np.uint8)
        self._ignore_value = ignore_value

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
    def segments(self):
        return self._segments
    
    @property
    def rgb_map(self):
        return self._rgb_map

    @property
    def framestamp(self):
        return self._frame.framestamp
    
    @property
    def timestamp(self):
        return self._frame.timestamp
    
    def set_frame(self,frame):
        self._frame.set_frame(frame)

    def mask_of(self,obj_name):
        a = np.zeros_like(self._segments)
        for i,obj in enumerate(self._obj_names):
            if obj_name in obj:
                a[self._segments == self._obj_ids[i]] = 1
                break
        return a
    
    def blend(self,frame=None,alpha=0.5, inplace=True):
        if frame is not None:
            # Assuming Falcoeye detection frame
            return frame.blend(self._rgb_map,alpha=alpha,inplace=inplace)
        else:
            return self._frame.blend(self._rgb_map,alpha=alpha,inplace=inplace)

    def keep_only(self,classes,inplace=True):
        a = np.zeros_like(self._segments, dtype=bool)
        _obj_ids = []
        _obj_names = []
        _colors = []
        logging.info(f"Keeping only {classes}")
        # looping over names of objects of interest
        for cl in classes:
            # loop over this segmentation objects
            for i,obj in enumerate(self._obj_names): 
                if cl in obj:
                    logging.info(f"Keeping {cl}")
                    # keep this object id, name and color
                    objid = self._obj_ids[i]
                    a[self._segments == objid] = 1
                    _obj_ids.append(objid)
                    _obj_names.append(obj)
                    _colors.append(self._colors[i])
                    break
        if inplace:
            
            self._segments[~a] = self._ignore_value
            logging.info(f"Keeping {np.unique(self._segments)} ids")
            # assuming ignore value will take 0,0,0 color
            self._rgb_map[~a] = [0,0,0]
            self._obj_ids = _obj_ids
            self._obj_names = _obj_names
            self._colors = _colors
            return self
        else:
            raise NotImplementedError
    
    def save_frame(self,path):
        Image.fromarray(self._frame).save(f"{path}/{self._frame_number}.png")

    def to_dict(self):
        dic = dict(
            frame=base64.b64encode(self._frame.frame).decode("utf-8"),
            obj_ids=[int(i) for i in self._obj_ids],
            obj_names=self._obj_names,
            colors=self._colors,
            segments = base64.b64encode(self._segments).decode("utf-8"),
            #rgb_maps=base64.b64encode(self._rgb_map).decode("utf-8"),
            ignore_value = int(self._ignore_value)
        )
        #logging.info(dic)
        return dic

    def __lt__(self,other):
        if type(other) == FalcoeyeSegmentation:
            return self._frame < other._frame
        else:
            # assuming other is FalcoeyeFrame
            return self._frame < other
    
    def __eq__(self,other):
        if type(other) == FalcoeyeSegmentation:
            return self._frame == other._frame
        else:
            # assuming other is FalcoeyeFrame or int
            return self._frame == other

class FalcoeyeSegmentationNode(Node):
    def __init__(self, name, 
    labelmap,ignore_value):
        Node.__init__(self,name)
        self._ignore_value = int(ignore_value)
        if type(labelmap) == str:
            with open(labelmap) as f:
                self._category_index = {int(k):v for k,v in json.load(f).items()}
        elif type(labelmap) == dict:
            self._category_index = {int(k):v for k,v in labelmap.items()}
    
    def translate(self):
        raise NotImplementedError
    
    def run(self):
        raise NotImplementedError
    
    def finalize(self,boxes,classes,scores):
        raise NotImplementedError
    
class FalcoeyeTorchSegmentationNode(FalcoeyeSegmentationNode):
    def __init__(self, name, labelmap,ignore_value):
        FalcoeyeSegmentationNode.__init__(self,name, labelmap,ignore_value)
        
    def translate(self,seg):
        seg_rgb = np.zeros((seg.shape[0], seg.shape[1], 3),dtype=np.uint8)
        objects = np.unique(seg)
        colors = []
        name_map = [self._category_index[int(obj)]["name"] for obj in objects]
        logging.info(f"These objects were found: {name_map}")
        for obj in objects:
            color = self._category_index[int(obj)]["color"]
            colors.append(color)
            seg_rgb[seg == obj] = color
        # seg might be read-only.. copy to force write
        return seg.copy(),objects,name_map,colors,seg_rgb
        
    def run(self):
        """
        Safe node: the input is assumed to be valid or can be handled properly, no need to catch
        """
        logging.info(f"Running falcoeye segmentation")
        while self.more():
            item = self.get()
            frame,raw_segmentation = item
            try:
                if type(raw_segmentation) == bytes:
                    raw_segmentation = np.frombuffer(raw_segmentation,dtype=np.float32)
                    raw_segmentation = raw_segmentation.reshape((frame.size[0],frame.size[1]))
                elif type(raw_segmentation) == list:
                    raw_segmentation = np.array(raw_segmentation)
                else:
                    # Acting safe
                    logging.warn("Couldn't parse the segmentation type. Returning zero array")
                    raw_segmentation = np.zeros((frame.size[0],frame.size[1]))
                
                seg,ids,names,colors,rgbs = self.translate(raw_segmentation)
                fe_segmentation = FalcoeyeSegmentation(frame,seg,ids,names,colors,rgbs,self._ignore_value)
                self.sink(fe_segmentation)
            except Exception as e:
                logging.error(e)
