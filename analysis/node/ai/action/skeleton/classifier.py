# -*- coding: utf-8 -*-
'''
This script includes:

TODO: Add more comments to this function.
'''
from ....node import Node

import os
import numpy as np
import pickle
from collections import deque
import cv2
from ...wrapper import FalcoeyeAIWrapper
from .feature_procs import FeatureGenerator
import logging
LABEL_UNKNOWN = ['', 0]

class PersonActionItem(FalcoeyeAIWrapper):
    def __init__(self,tracker):
        FalcoeyeAIWrapper.__init__(self,
            tracker._frame)
        self._tracker = tracker
        if self._tracker is None:
            self._actions = []
        else:
            self._actions = [LABEL_UNKNOWN]*len(self._tracker)

    def set_label(self,index,label):
        self._actions[index] = label
    
    def draw(self,actions=True):
        # to skip drawing tracking and draw it here
        self._tracker.draw(tracking=False)
        if not actions:
            return
        ids = self._tracker.get_tracked_ids()
        logging.info(f"Drawing action for ids {ids}")
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
        for index,hid in enumerate(ids):
            track_label = f'{hid} {self._actions[index][0]}'
            x1, y1, x2, y2 = self._tracker.get_box(index).astype(np.int16)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
            track_loc = get_label_position(track_label)
            logging.info(f"Plotting {track_label} {x1},{y1},{track_loc[0]},{track_loc[1]},{track_loc[2]}")
            cv2.rectangle(frame, (x1, y1), (track_loc[0], track_loc[1]), (255, 0, 0), -1)
            cv2.putText(frame, track_label, (x1+1, track_loc[2]), font,
                    scale,(0, 0, 0), thickness)
        self.set_frame(frame)

class PersonClassifier(object):
    ''' Classifier for online inference.
        The input data to this classifier is the raw skeleton data, so they
            are processed by `class FeatureGenerator` before sending to the
            self.model trained by `class ClassifierOfflineTrain`.
    '''

    def __init__(self, model_node, action_labels, window_size, human_id=0, threshold=0.3):
        
        self._model = model_node
        # -- Settings
        self.human_id = human_id
        
        self.action_labels = action_labels
        self.threshold = threshold

        # -- Time serials storage
        self.feature_generator = FeatureGenerator(window_size)
        self.reset()

    def reset(self):
        self.feature_generator.reset()
        self.scores_hist = deque()
        self.scores = None

    def predict(self, skeleton):
        ''' Predict the class (string) of the input raw skeleton '''
        
        is_features_good, features = self.feature_generator.add_cur_skeleton(skeleton)

        if is_features_good:
            # convert to 2d array
            features = features.reshape(-1, features.shape[0])
            #logging.info(f"Features shape: {features.shape}")
            # this should return buffer with array of size (1,n_labels)
            curr_scores = self._model(features,as_image=False)
            # the size of this should be (n_labels,) since b_size = 1 and gRPC will 
            # return flat array
            curr_scores = np.frombuffer(curr_scores,dtype=np.float32)
            #logging.info(f"{curr_scores},{curr_scores.shape}")
            self.scores = self.smooth_scores(curr_scores)
            

            if self.scores.max() < self.threshold:  # If lower than threshold, bad
                prediced_label = LABEL_UNKNOWN
            else:
                predicted_idx = self.scores.argmax()
                prediced_label = self.action_labels[predicted_idx], self.scores.max()

        else:
            prediced_label = LABEL_UNKNOWN
        return prediced_label

    def smooth_scores(self, curr_scores):
        ''' Smooth the current prediction score
            by taking the average with previous scores
        '''
        self.scores_hist.append(curr_scores)
        DEQUE_MAX_SIZE = 2
        if len(self.scores_hist) > DEQUE_MAX_SIZE:
            self.scores_hist.popleft()

     
        score_sums = np.zeros((len(self.action_labels),))
        for score in self.scores_hist:
            score_sums += score
        score_sums /= len(self.scores_hist)
        # print("\nMean score:\n", score_sums)
        return score_sums

        

class MultiPersonClassifier(Node):
    ''' This is a wrapper around ClassifierOnlineTest
        for recognizing actions of multiple people.
    '''

    def __init__(self, 
        name,
        model_node, 
        classes, 
        window_size=5, 
        threshold=0.3):
        Node.__init__(self,name)
        self._model_node = model_node
        self._classes = classes
        self._window_size = window_size
        self._threshold = threshold
        self.dict_id2clf = {}  # human id -> classifier of this person
        
        # Define a function for creating classifier for new people.
        self._create_classifier = lambda human_id: PersonClassifier(
            model_node, classes, window_size, human_id, threshold=threshold)

    def classify(self, tr_item):
        ''' Classify the action type of each skeleton in dict_id2skeleton '''
        action_item = PersonActionItem(tr_item)

        if tr_item is None or len(tr_item) == 0:
            return action_item

        dict_id2skeleton = {tr_item.get_tracked_id(index): 
            tr_item.get_flatten_keypoints(index) for index in range(len(tr_item))}
        # Clear people not in view
        old_ids = set(self.dict_id2clf)
        cur_ids = set(dict_id2skeleton)
        humans_not_in_view = list(old_ids - cur_ids) # check person is missed or not
        for human in humans_not_in_view:
            del self.dict_id2clf[human]

        # Predict each person's action
        for index,tr_id in enumerate(tr_item.get_tracked_ids()):
            if tr_id not in self.dict_id2clf:  # add this new person
                self.dict_id2clf[tr_id] = self._create_classifier(id)

            # getting classifier
            classifier = self.dict_id2clf[tr_id]
            # classify
            label_name,label_index = classifier.predict(dict_id2skeleton[tr_id])
            # set label
            action_item.set_label(index,(label_name,label_index))

        return action_item

    def run(self):
        # expecting items of type FalcoeyeFrame or a wrapper for it
        while self.more():
            item = self.get()
            # assuming FalcoeyeTracker or a wrapper for it
            logging.info(f"Running {self._name} on item {item.framestamp}")
            cls_item = self.classify(item)
            self.sink(cls_item)
