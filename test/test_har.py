from .utils import load_workflow_structure,get_video,get_prefix
from .utils import mocked_finish_put
from unittest import mock
import os
import sys
from ..analysis.workflow.workflow import WorkflowFactory
import logging


@mock.patch("requests.put", side_effect=mocked_finish_put)
def test_har(mock_put):
    logging.info("Launching test_har")
    wf = load_workflow_structure("human_action_recognition")
    assert wf
    video = get_video("ocean_city_1")
    assert video
    prefix = get_prefix("test_har")
    analysis = {
        "id":"test_har",
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