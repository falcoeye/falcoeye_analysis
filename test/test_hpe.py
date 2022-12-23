from .utils import load_workflow_structure,get_video,get_prefix
from .utils import mocked_finish_put
from unittest import mock
import os
import sys
from ..analysis.workflow.workflow import WorkflowFactory
import logging

@mock.patch("requests.put", side_effect=mocked_finish_put)
def test_hpe(mock_put):
    logging.info("Launching test_hpe")
    wf = load_workflow_structure("human_pose")
    assert wf
    video = get_video("ocean_city_1")
    assert video
    prefix = get_prefix("test_hpe")
    analysis = {
        "id":"test_hpe",
        "args": {
            "filename": video,
            "length": 30*10,
            "sample_every": 10,
            "frequency": 4,
            "ntasks": 4,
            "prefix": prefix
        }
    }
    
    # Creating workflow handler
    workflow = WorkflowFactory.create_from_dict(wf,analysis)
    logging.info("Starting the workflow")
    workflow.run_sequentially_async()

    while workflow._busy:
        pass

@mock.patch("requests.put", side_effect=mocked_finish_put)
def test_with_tracking_hpe(mock_put):
    logging.info("Launching test_with_tracking_hpe")
    wf = load_workflow_structure("human_pose_with_tracking")
    assert wf
    video = get_video("ocean_city_1")
    assert video
    prefix = get_prefix("test_hpe_with_tra")
    analysis = {
        "id":"test_with_tracking_hpe",
        "args": {
            "filename": video,
            "length": 30*60,
            "sample_every": 3,
            "frequency": 30,
            "ntasks": 8,
            "prefix": prefix
        }
    }
    
    # Creating workflow handler
    workflow = WorkflowFactory.create_from_dict(wf,analysis)
    logging.info("Starting the workflow")
    workflow.run_sequentially_async()

    while workflow._busy:
        pass