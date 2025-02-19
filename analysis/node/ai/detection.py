
from ..node import Node
import json
import logging
import numpy as np
from .utils import get_color_from_number,non_max_suppression
from PIL import ImageDraw, Image
from .wrapper import FalcoeyeAIWrapper
import cv2

class FalcoeyeDetection(FalcoeyeAIWrapper):
    def __init__(self,frame,detections, category_map,width_height=False):
        FalcoeyeAIWrapper.__init__(self,frame)
        self._detections = detections
        self._category_map = {c:len(v) for c,v in category_map.items()}

        self._boxes = [d["box"] for d in self._detections]
        self._classes = [d["class"] for d in self._detections]

    @property
    def count(self):
        return len(self._detections)
    
    @property
    def boxes(self):
        return self._boxes

    @property
    def classes(self):
        return self._classes

    def translate_pixel(self, x, y):
        height,width,_ = self.size
        return int(x * width), int(y * height)

    def iwidth(self,i):
        return np.abs(self._boxes[i][0]-self._boxes[i][2])
    
    def iheight(self,i):
        return np.abs(self._boxes[i][1]-self._boxes[i][3])

    def count_of(self, category):
        if category in self._category_map:
            return self._category_map[category]
        return -1

    def get_class_instances(self, name):
        return [i for i, d in enumerate(self._detections) if d["class"] == name]

    def get_class(self, i):
        return self._detections[i]["class"]

    def get_box(self, i):
        return self._detections[i]["box"]

    def keep_only(self,keys,inplace=True):
        # TODO: maybe optimize in one loop 
        if inplace:
            n = self.count
            index = 0
            for _ in range(n):
                cl = self.get_class(index)
                if cl not in keys:
                    self.delete(index)
                else:
                    # when deleting, no index increament due to shift in array
                    index += 1
        else:
            raise NotImplementedError
    
    def delete(self,index):
        item = self._detections.pop(index)
        self._category_map[item["class"]] -= 1
        self._boxes.pop(index)
        self._classes.pop(index)
    
    def blend(self,image,alpha=0.5,inplace=True):
        return self._frame.blend(image,alpha,inplace)

    def draw_bounding_box(self,index,color,inplace=True,translate=True):
        if inplace:
            xmin, ymin, xmax, ymax = self.get_box(index)
            if translate:
                xmin, ymin = self.translate_pixel(xmin * 0.9999, ymin * 0.9999)
                xmax, ymax = self.translate_pixel(xmax * 0.9999, ymax * 0.9999)
            img = self._frame.frame.copy()
            thickness = 2
            scale = 0.6 
            font = cv2.FONT_HERSHEY_COMPLEX
            def get_label_position(label):  
                w, h = cv2.getTextSize(label, font, scale, thickness)[0]
                offset_w, offset_h = w + 3, h + 5
                xmax = x1 + offset_w
                ymax = y1 + offset_h
                y_text = ymax - 2
                return xmax, ymax, y_text
            
            label = self.get_class(index)
            x1, y1, x2, y2 = int(xmin), int(ymin), int(xmax), int(ymax)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            label_loc = get_label_position(label)
            cv2.putText(img, label, (x1+1, label_loc[2]), font,
                    scale,(0, 0, 0), thickness)
            self._frame.set_frame(img)
        else:
            raise NotImplementedError

    def __lt__(self,other):
        if type(other) == FalcoeyeDetection:
            return self._frame < other._frame     
        return FalcoeyeAIWrapper.__lt__(self,other)
    
    def __eq__(self,other):
        if type(other) == FalcoeyeDetection:
            return self._frame == other._frame
        return FalcoeyeAIWrapper.__eq__(self,other)

class FalcoeyeDetectionNode(Node):
    def __init__(self, name, 
    labelmap,
    min_score_thresh,
    max_boxes=100,
    overlap_thresh=0.3):
        Node.__init__(self,name)
        self._min_score_thresh = min_score_thresh
        self._max_boxes = max_boxes
        self._overlap_thresh = overlap_thresh
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
        _detections = []
        _category_map = {k:[] for k in self._category_index.values()}
        _category_map["unknown"] = []
        # TODO: can be optimized with nms suppresion
        conf_mask = np.where(scores>self._min_score_thresh)
        logging.info(f"Conf mask {conf_mask[0].shape} min score thresh {self._min_score_thresh} score min {scores.min()}")
        boxes = boxes[conf_mask]
        classes = classes[conf_mask]
        scores = scores[conf_mask]

        logging.info(f"Applying non-max suppression with threshold {self._overlap_thresh} on {boxes.shape[0]} items")
        if self._overlap_thresh:
            nms_picks = non_max_suppression(
                boxes,scores,self._overlap_thresh
            )
        else:
            nms_picks = range(boxes.shape[0])
        logging.info(f"Number of items after non-max suppression is {len(nms_picks)}")
        for counter,p in enumerate(nms_picks):
            if classes[p] in self._category_index:
                class_name = str(self._category_index[classes[p]])
            else:
                class_name = "unknown"
            color = get_color_from_number(int(classes[p]))
            _detections.append(
                {
                    "box": tuple(boxes[p].tolist()),
                    "color": color,
                    "class": class_name,
                    "score": round(scores[p], 2) * 100,
                }
            )
            _category_map[class_name].append(counter)
        return _detections, _category_map

class FalcoeyeTFDetectionNode(FalcoeyeDetectionNode):
    def __init__(self, name, 
    labelmap,
    min_score_thresh,
    max_boxes,
    overlap_thresh=0.3):
        FalcoeyeDetectionNode.__init__(self,name, 
        labelmap,
        min_score_thresh,
        max_boxes,
        overlap_thresh)

    def translate(self,detections):
        logging.info("Translating detection")
        if detections is None or type(detections) != dict or "detection_boxes" not in detections:
            return _detections, _category_map
        
        boxes = np.array(detections["detection_boxes"])
        # y1,x1,y2,x2 --> x1, y1, x2, y2
        boxes = boxes[:,[1,0,3,2]]
        classes = np.array(detections["detection_classes"]).astype(int)
        scores = np.array(detections["detection_scores"])
        
        logging.info(f"#boxes {boxes.shape}, #classes {classes.shape} #scores {scores.shape}")
        return self.finalize(boxes,classes,scores)
        
       
    def run(self):
        """
        Safe node: the input is assumed to be valid or can be handled properly, no need to catch
        """
        logging.info(f"Running falcoeye detection")
        while self.more():
            item = self.get()

            frame,raw_detections = item
            try:
                if type(raw_detections) == dict and "prediction" in raw_detections:
                    # restful
                    raw_detections = raw_detections['predictions'][0]
                elif raw_detections is None:
                    # TODO: should do failover
                    raw_detections = {'detection_boxes': np.array([]),
                    'detection_classes':np.array([]),
                    'detection_scores': np.array([])}
                else:
                    # gRPC response
                    logging.info("gRPC detection")
                    boxes = np.array(raw_detections.outputs['detection_boxes'].float_val).reshape((-1,4)).tolist()
                    classes = raw_detections.outputs['detection_classes'].float_val
                    scores = raw_detections.outputs['detection_scores'].float_val
                    raw_detections = {'detection_boxes': boxes,
                        'detection_classes':classes,
                        'detection_scores': scores}
                
                logging.info(f"New frame for falcoeye detection  {frame.framestamp} {frame.timestamp}")
                detections, category_map = self.translate(raw_detections)
                fe_detection = FalcoeyeDetection(frame,detections, 
                        category_map)
                self.sink(fe_detection)
            except Exception as e:
                logging.error(e)
    
class FalcoeyeTorchDetectionNode(FalcoeyeDetectionNode):
    def __init__(self, name, 
        labelmap,
        min_score_thresh,
        max_boxes,
        overlap_thresh=0.3,
        width_height_box=False):
        FalcoeyeDetectionNode.__init__(self,name, labelmap,
        min_score_thresh,
        max_boxes,
        overlap_thresh)
        self._width_height_box = width_height_box
        

    def translate(self,detections):
        bboxes = detections[..., :4].reshape(-1,4)
        if self._width_height_box:
            # copy because it is read-only
            bboxes = bboxes.copy()
            bboxes[:,2] = bboxes[:,2]+bboxes[:,0]
            bboxes[:,3] = bboxes[:,3]+bboxes[:,1]
        scores = detections[..., 4]
        classes = detections[..., 5]
        return self.finalize(bboxes,classes,scores)
        
    def run(self):
        """
        Safe node: the input is assumed to be valid or can be handled properly, no need to catch
        """
        logging.info(f"Running falcoeye detection")
        while self.more():
            item = self.get()
            frame,raw_detections = item
            try:
                if type(raw_detections) == bytes:
                    raw_detections = np.frombuffer(raw_detections,dtype=np.float32).reshape(-1,6)
                elif type(detections) == list:
                    raw_detections = np.array(raw_detections)
                else:
                    # Acting safe
                    raw_detections = {'detection_boxes': np.array([]),
                            'detection_classes':np.array([]),
                            'detection_scores': np.array([])}
                
                detections, category_map = self.translate(raw_detections)
                fe_detection = FalcoeyeDetection(frame,detections, 
                            category_map)
                self.sink(fe_detection)
            except Exception as e:
                logging.error(e)

class FalcoeyeTritonDetectionNode(FalcoeyeDetectionNode):
    def __init__(self, name, 
        labelmap,
        min_score_thresh,
        max_boxes,
        overlap_thresh=0.3):
        FalcoeyeDetectionNode.__init__(self, name,
            labelmap,
            min_score_thresh,
            max_boxes,
            overlap_thresh)
        
    def translate(self, detections):
        """Translates Triton server output format to the standard detection format"""
        logging.info("Translating Triton detection output")
        
        if detections is None or not isinstance(detections, dict):
            logging.warning(f"Invalid detection format received: {type(detections)}")
            return [], {k:[] for k in self._category_index.values()}
        
        try:
            # Get the number of detections (should be a scalar)
            num_dets = int(detections['num_dets'].flatten()[0])
            logging.info(f"Processing {num_dets} detections")
            
            # Extract detections from the first batch
            boxes = detections['det_boxes'][0, :num_dets]
            scores = detections['det_scores'][0, :num_dets]
            classes = detections['det_classes'][0, :num_dets].astype(int)
            
            return self.finalize(boxes, classes, scores)
            
        except Exception as e:
            logging.error(f"Error processing detections: {str(e)}")
            traceback.print_exc()  # Add stack trace for debugging
            return [], {k:[] for k in self._category_index.values()}
    
    def run(self):
        """
        Processes detection results from Triton server output
        """
        logging.info("Running Triton detection node")
        while self.more():
            item = self.get()
            frame, raw_detections = item
            
            try:
                # Handle different input formats
                if isinstance(raw_detections, dict) and "outputs" in raw_detections:
                    # Handle gRPC response format
                    detections = {
                        "num_dets": np.array(raw_detections.outputs["num_dets"].as_numpy()),
                        "det_boxes": np.array(raw_detections.outputs["det_boxes"].as_numpy()),
                        "det_scores": np.array(raw_detections.outputs["det_scores"].as_numpy()),
                        "det_classes": np.array(raw_detections.outputs["det_classes"].as_numpy())
                    }
                elif isinstance(raw_detections, dict) and all(key in raw_detections for key in ["num_dets", "det_boxes", "det_scores", "det_classes"]):
                    # Already in the correct format
                    detections = raw_detections
                else:
                    # Handle invalid input
                    logging.warning(f"Invalid detection format received: {type(raw_detections)}")
                    detections = {
                        "num_dets": np.array([0]),
                        "det_boxes": np.array([]),
                        "det_scores": np.array([]),
                        "det_classes": np.array([])
                    }
                
                logging.info(f"Processing frame for Triton detection: {frame.framestamp} {frame.timestamp}")
                detections, category_map = self.translate(detections)
                fe_detection = FalcoeyeDetection(frame, detections, category_map)
                self.sink(fe_detection)
                
            except Exception as e:
                logging.error(f"Error processing Triton detection: {str(e)}")
                continue