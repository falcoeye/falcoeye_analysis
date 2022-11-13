

from .utils import load_workflow_structure,get_video,get_prefix
from .utils import mocked_finish_put
from unittest import mock
import os
import sys
from ..analysis.workflow.workflow import WorkflowFactory
import logging
import requests
import time
import json


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

def test_inline_backend_table_finder(backend,test_user_token,analysis,workflows,videos):
    logging.info("Launching inline_table_finder by calling the backend")
    header = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "X-API-KEY": test_user_token
    }
    wfid = None
    for wf in workflows:
        if wf["name"] == "Tabel Finder":
            wfid = wf["id"]
            break
    assert wfid
    logging.info(f"Found table finder workflow {wfid}")
    
    vidid = None
    for vid in videos:
        if "table" in vid["tags"]:
            vidid = vid["id"]
            break
    assert vidid
    
    for anl in analysis:
        if "test_inline_backend_table_finder"  ==  anl["name"]:
            anlid = anl["id"]
            logging.info("Found analysis with same name")
            resp = requests.delete(f"{backend}/api/analysis/{anlid}", headers=header)
            assert resp.status_code == 200
            break

    args = {
        "workflow_id": wfid,
        "feeds": {
            "source": {
                "id": vidid,
                "type": "video"
            },
            "params": {

            }
        }
    }
    logging.info(f"Posting new inline analysis with args: {args}")
    resp = requests.post(f"{backend}/api/analysis/", json=args, headers=header)
    logging.info(f"Post request status: {resp.status_code}")
    resdict = resp.json()
    message = resdict["message"]
    assert message == "analysis added"
    analysis_id = resdict["analysis"]["id"]
    logging.info(f"Analysis created {analysis_id}")
    resp = requests.get(f"{backend}/api/analysis/{analysis_id}/meta.json", headers=header)
    while resp.status_code == 204:
        logging.info("not yet")
        time.sleep(1)
        resp = requests.get(f"{backend}/api/analysis/{analysis_id}/meta.json", headers=header)

    logging.info(f"Meta is found: {resp.status_code}")
    meta = json.loads(resp.content.decode("utf-8"))
    ffile = meta["filenames"][0]
    resp = requests.get(f"{backend}/api/analysis/{analysis_id}/{ffile}", headers=header)
    data = json.loads(resp.content.decode("utf-8"))
    logging.info(f"Data has keywords {data.keys()}")
    logging.info(f"Object names {data['obj_names']}")
    assert "table" in [i for d in data["obj_names"] for i in d]


    
