

from .utils import load_workflow_structure,get_video,get_prefix
from .utils import mocked_finish_put
from unittest import mock
import os
import sys
from ..analysis.workflow.workflow import WorkflowFactory
import logging

@mock.patch("requests.put", side_effect=mocked_finish_put)
def test_inline_table_finder(mock_put):
    logging.info("Launching inline_table_finder")
    wf = load_workflow_structure("table_finder",inline=True)
    assert wf
    video = get_video("sketchen")
    assert video
    prefix = get_prefix("inline_table_finder")
    analysis = {
        "id":"inline_table_finder",
        "args": {
            "filename": video,
            "prefix": prefix
        }
    }
    
    # Creating workflow handler
    workflow = WorkflowFactory.create_from_dict(wf,analysis)
    logging.info("Starting the workflow")
    workflow.run_sequentially()