
from .utils import load_workflow_structure,get_video,get_prefix
from .utils import mocked_finish_put
from unittest import mock
import os
import sys
from ..analysis.workflow.workflow import WorkflowFactory
import logging

@mock.patch("requests.put", side_effect=mocked_finish_put)
def test_segmenter_video(mock_put):
    logging.info("Launching test_segmenter_video")
    wf = load_workflow_structure("segmenter")
    assert wf
    video = get_video("coffeeshop_brew")
    assert video
    prefix = get_prefix("segmenter_video")
    analysis = {
        "id":"test_segmenter_video",
        "args": {
            "filename": video,
            "length": 180,
            "sample_every": 1,
            "frequency": 5,
            "timed_gate_open_freq": 30,
            "timed_gate_opened_last": 30,
            "ntasks": 4,
            "prefix": prefix,
            "size": "640,360"
        }
    }
    
    # Creating workflow handler
    workflow = WorkflowFactory.create_from_dict(wf,analysis)
    logging.info("Starting the workflow")
    workflow.run_sequentially_async()

    while workflow._busy:
        pass