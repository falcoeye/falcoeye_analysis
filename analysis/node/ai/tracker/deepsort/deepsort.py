import cv2
import logging
import numpy as np
from ....node import Node
from ...wrapper import FalcoeyeAIWrapper
from . import linear_assignment
from . import iou_matching
from . import kalman_filter
from .track import Track
from .detection import Detection
from .nn_matching import NearestNeighborDistanceMetric

class TrackingItem(FalcoeyeAIWrapper):
    def __init__(self,aiwrapper):
        FalcoeyeAIWrapper.__init__(self,aiwrapper._frame)
        self._aiwrapper = aiwrapper
        if self._aiwrapper is None:
            self._ids = []
        else:
            self._ids = [-1]*len(self._aiwrapper)
    
    def set_tracked_id(self,index,id):
        self._ids[index] = id
    
    def get_tracked_id(self,index):
        return self._ids[index]

    def get_tracked_ids(self):
        return self._ids

    def delete_non_tracked(self):
        non_tracked = [i for i,v in enumerate(self._ids) if v==-1]
        for index in sorted(non_tracked, reverse=True):
            del self._ids[index]
        self._aiwrapper.delete(non_tracked)
    
    def __len__(self):
        return len(self._aiwrapper)
    
    def get_boxes(self):
        return self._aiwrapper.get_boxes()
    
    def get_box(self,index):
        return self._aiwrapper.get_box(index)

    def set_box(self,index,box):
        self._aiwrapper.set_box(index,box)

    def get_flatten_keypoints(self,index):
        return self._aiwrapper.get_flatten_keypoints(index)

    def draw(self,tracking=True):
        logging.info(f"Drawing tracked ids {self._ids}")
        self._aiwrapper.draw()
        if not tracking:
            return 

        thickness = 1
        scale = 0.6 
        font = cv2.FONT_HERSHEY_COMPLEX
        def get_label_position(label):  
            w, h = cv2.getTextSize(label, font, scale, thickness)[0]
            offset_w, offset_h = w + 3, h + 5
            xmax = x1 + offset_w
            ymax = y1 + offset_h
            y_text = ymax - 2
            return xmax, ymax, y_text
        frame = self.frame.copy()
        for index,hid in enumerate(self._ids):
            track_label = f'{hid}'
            x1, y1, x2, y2 = self._aiwrapper.get_box(index).astype(np.int16)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
            track_loc = get_label_position(track_label)
            logging.info(f"Plotting {track_label} {x1},{y1},{track_loc[0]},{track_loc[1]},{track_loc[2]}")
            cv2.rectangle(frame, (x1, y1), (track_loc[0], track_loc[1]), (255, 0, 0), -1)
            cv2.putText(frame, track_label, (x1+1, track_loc[2]), font,
                    scale,(0, 0, 0), thickness)
        self.set_frame(frame)


class DeepSortTracker(Node):
    
    def __init__(self,
        name,
        feature_extractor_node,
        max_dist=0.2, 
        max_iou_distance=0.7, 
        max_age=70, 
        n_init=3, 
        nn_budget=100
        ):
        Node.__init__(self,name)
        self.extractor = feature_extractor_node
        self.metric = NearestNeighborDistanceMetric(
            "cosine",
            max_dist,
            nn_budget)
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def _get_features(self, bbox_tlbr, ori_img):
        features = []
        #logging.info("Getting feature")
        for box in bbox_tlbr:
            x1, y1, x2, y2 = map(int, box)
            im = ori_img[y1:y2, x1:x2]
            ifeatures = self.extractor(im)
            ifeatures = np.frombuffer(ifeatures,dtype=np.float32)
            features.append(ifeatures)
        features = np.array(features)
        #logging.info(f"Features shape {features.shape}")
        return features

    def _predict(self):
        """Propagate track state distributions one time step forward.
        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)
    
    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]

        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]

        matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        # matches = matches_b
        # unmatched_tracks = list(set(unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1

    def predict(self, aiwrapper):
        """Update tracker state via analyis of current keypoint's bboxes with previous tracked bbox.
        args:
            aiwrapper (FalcoeyeAIWrapper): an ai model prediction wrapper with keypoints bboxes, (xmin, ymin, w, h).
        return:
            tracked_predictions (list): Filtered tracked list of annotations object filled with
                    tracked id, tracked color and bbox (top,left,btm,right) attributes.
        """

        # generate detections: assume of type FalcoeyeAIWrapper 
        # or its children
        tr_item = TrackingItem(aiwrapper)
        if len(tr_item) == 0:
            return tr_item
         
        bbox_tlwh = np.array(tr_item.get_boxes())
        bbox_tlbr = self.tlwh_to_tlbr(bbox_tlwh)

        features = self._get_features(bbox_tlbr, tr_item.frame)
        
        detections = [Detection(bbox, features[i]) for i, bbox in enumerate(bbox_tlwh)]

        # update tracker and predictions object
        self._predict() # update track_id's time_since_update and age increasement
        self.update(detections, tr_item) # update predictions with tracked ID and Color
        # filter untracked persons' keypoints
        tr_item.delete_non_tracked()
    
        return tr_item
    
    def increment_ages(self):
        for track in self.tracks:
            track.increment_age()
            track.mark_missed()

    def update(self, detections, tracking_item):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
        predictions : List[annoation.Annotation]
            A list of annoations object to update the tracked person info.
        """


        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)
        # Update track set.
        for track_idx, pred_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[pred_idx])
            track_id = self.tracks[track_idx].track_id
            track_box_tlbr = self.tracks[track_idx].to_tlbr()
            # update track_id, tracked_color, tlbr_bbox of predictions object
            tracking_item.set_tracked_id(pred_idx,track_id)
            tracking_item.set_box(pred_idx,track_box_tlbr)

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        features, targets, active_targets = [], [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            active_targets.append(track.track_id)
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    @staticmethod
    def tlwh_to_tlbr(bbox_tlwh):
        if isinstance(bbox_tlwh, np.ndarray):
            bbox_tlbr = bbox_tlwh.copy()
        elif isinstance(bbox_tlwh, torch.Tensor):
            bbox_tlbr = bbox_tlwh.clone()

        bbox_tlbr[:, 2] += bbox_tlwh[:, 0]
        bbox_tlbr[:, 3] += bbox_tlwh[:, 1]
        return bbox_tlbr

    def run(self):
        # expecting items of type FalcoeyeFrame or a wrapper for it
        while self.more():
            item = self.get()
            # assuming FalcoeyeOpenPoseHPE or a wrapper for it
            logging.info(f"Running {self._name} on item {item.framestamp}")
            tr_item = self.predict(item)

            self.sink(tr_item)
