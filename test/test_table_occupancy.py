from .utils import load_workflow_structure,get_video,get_prefix
from .utils import mocked_finish_put
from unittest import mock
import os
import sys
from ..analysis.workflow.workflow import WorkflowFactory
import logging

@mock.patch("requests.put", side_effect=mocked_finish_put)
def test_table_occupancy(mock_put):
    logging.info("Launching test_table_occupancy")
    wf = load_workflow_structure("table_occupancy")
    assert wf
    video = get_video("sketchen")
    assert video
    prefix = get_prefix("table_occupancy")
    analysis = {
        "id":"test_table_occupancy",
        "args": {
            "filename": video,
            "length": -1,
            "sample_every": 300,
            "frequency": 3,
            "ntasks": 4,
            "prefix": prefix,
            "min_score_thresh": 0.20,
            "tables": "Table 1:1013,316,878,399,1039,559,1147,422;Tabel 2:826,158,663,246,832,386,972,261;Tabel 3:651,28,500,111,649,242,831,125;Tabel 4:180,444,373,344,266,209,119,283;Tabel 5:382,369,173,478,284,718,534,572"
        }
    }
    
    # Creating workflow handler
    workflow = WorkflowFactory.create_from_dict(wf,analysis)
    logging.info("Starting the workflow")
    workflow.run_sequentially_async()

    while workflow._busy:
        pass