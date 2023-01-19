from .utils import load_workflow_structure,get_video,get_prefix
from .utils import mocked_finish_put
from unittest import mock
import os
import sys
from ..analysis.workflow.workflow import WorkflowFactory
import logging

@mock.patch("requests.put", side_effect=mocked_finish_put)
def test_road_damage(mock_put):
    logging.info("Launching road damge")
    wf = load_workflow_structure("road_damage_4m")
    assert wf
    video = get_video("riyadh_st")
    assert video
    prefix = get_prefix("road_damage")
    analysis = {
        "id":"test_road_damage",
        "args": {
            "filename": video,
            "length": -1,
            "sample_every": 500,
            "frequency": 3,
            "ntasks": 4,
            "prefix": prefix,
            "min_score_thresh": 0.10,
            "gps_info": "1570,1853,2382,1950"
        }
    }
    
    # Creating workflow handler
    workflow = WorkflowFactory.create_from_dict(wf,analysis)
    logging.info("Starting the workflow")
    workflow.run_sequentially_async()

    while workflow._busy:
        pass

@mock.patch("requests.put", side_effect=mocked_finish_put)
def test_nogps_road_damage(mock_put):
    logging.info("Launching road damge")
    wf = load_workflow_structure("road_damage_4m_nogps")
    assert wf
    video = get_video("roaddamage")
    assert video
    prefix = get_prefix("road_damage_4m_nogps")
    analysis = {
        "id":"test_road_damage_4m_nogps",
        "args": {
            "filename": video,
            "length": -1,
            "sample_every": 30,
            "frequency": 3,
            "ntasks": 4,
            "prefix": prefix,
            "min_score_thresh": 0.30
        }
    }
    
    # Creating workflow handler
    workflow = WorkflowFactory.create_from_dict(wf,analysis)
    logging.info("Starting the workflow")
    workflow.run_sequentially_async()

    while workflow._busy:
        pass