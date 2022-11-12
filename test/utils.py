import json
import os
import glob
from unittest import mock
import logging

basedir = os.path.abspath(os.path.dirname(__file__))

WORKFLOWS = {
   os.path.basename(f).replace(".json",""): f for f in glob.glob(f"{basedir}/../../falcoeye_backend/initialization/workflows/*.json")
}
INLINE_WORKFLOWS = {
   os.path.basename(f).replace(".json",""): f for f in glob.glob(f"{basedir}/../../falcoeye_backend/initialization/inline_workflows/*.json")
}
logging.info(INLINE_WORKFLOWS)
VIDEOS = {
    os.path.basename(f).replace(".mp4",""): f for f in glob.glob(f"{basedir}/../../media/*.mp4")
}
def load_workflow_structure(name,inline=False):
    if not inline and name not in WORKFLOWS:
        logging.info("Couldn't find async workflow")
        return None
    elif inline and name not in INLINE_WORKFLOWS:
        logging.info("Couldn't find inline workflow")
        return None
    
    if inline:
        wf = INLINE_WORKFLOWS[name]
    else:
        wf = WORKFLOWS[name]

    with open(wf) as f:
        data = f.read()
    structure = json.loads(data)["structure"]
    return structure

def get_video(name):
    if name not in VIDEOS:
        return None
    return VIDEOS[name]

def mkdir(path):
    path = os.path.relpath(path)
    makedirs(path)

def get_prefix(name):
    path = f"{basedir}/outputs/{name}/"
    os.makedirs(path,exist_ok = True)
    return path

def mocked_finish_put(*args, **kwargs):
    logging.info("Mocked post called")
    class MockResponse:
        def __init__(self, json_data, status_code):
            self.json_data = json_data
            self.status_code = status_code
            self.headers = {"content-type":"application/json"}

        def response(self):
            return self.json_data, self.status_code

        def json(self):
            return self.json_data

    mockresponse =  MockResponse({"status": True, "message": "Good for you"}, 200)
    logging.info("Mocked response created")
    return mockresponse
