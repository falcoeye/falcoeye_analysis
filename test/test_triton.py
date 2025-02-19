
from .utils import load_workflow_structure,get_video,get_prefix
from .utils import mocked_finish_put
from unittest import mock
import os
import sys
from ..analysis.workflow.workflow import WorkflowFactory
import logging

@mock.patch("requests.put", side_effect=mocked_finish_put)
def test_triton(mock_put):
    logging.info("Launching test_triton")
    wf = load_workflow_structure("kaust_fish_counter_yolovfish")
    assert wf
    video = get_video("arabian_angelfish_short")
    assert video
    prefix = get_prefix("triton_video")
    analysis = {
        "id":"test_triton",
        "args": {
            "filename": video,
            "length": 60,
            "sample_every": 1,
            "frequency": 1,
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