from .utils import fill_args
from ..node import create_node_from_dict,Source
import logging 
import threading
from ..utils import get_service,message as ResponseMessage
import requests
import json
import os

def post_new_status(aid,status,msg):
    try:
        resp = { "status": status }
        resp["message"] = msg
        
        backend_server = get_service("falcoeye-backend")
        postback_url = f"{backend_server}/api/analysis/{aid}"
        logging.info(f"Posting new status {status} to backend {postback_url}")
        rv = requests.put(
            postback_url,
            data=json.dumps(resp),
            headers={"Content-type": "application/json","X-API-KEY":os.environ.get("JWT_KEY")},
        )
        if rv.headers["content-type"].strip().startswith("application/json"):
            logging.info(f"Response received {rv.json()}")

        else:
            logging.info(f"Request might have failed. No json response received")

    except requests.exceptions.ConnectionError:
        logging.error(
            f"Warning: failed to inform backend server ({backend_server}) for change in the status "
        )
    except requests.exceptions.Timeout:
        logging.error(
            f"Warning: failed to inform backend server ({backend_server}) for change in the status "
        )
    except requests.exceptions.HTTPError:
        logging.error(
            f"Warning: failed to inform backend server ({backend_server}) for change in the status "
        )

class Workflow:
    def __init__(self,analysis_id,nodes,starters,nodes_in_order):
        self._analysis_id = analysis_id
        self._nodes = nodes
        self._starters = starters
        self._nodes_in_order = nodes_in_order
        self._busy = False
        self._tasks = {}
    
    def run_sequentially(self):
        logging.info(f"Running {self._analysis_id} sequentially")
        self._busy = True
        for n in self._nodes_in_order:
            logging.info(f"Running {n._name}")
            n.run()
        logging.info(f"{self._analysis_id} completed")
        self._busy = False

    def run_sequentially_async(self):
        logging.info(f"Running {self._analysis_id} asynchronously")
        self._busy = True
        self._tasks = {}
        for n in self._nodes_in_order:
            # TODO: rename or do something, looks ugly
            self._tasks[n.name] = {"done":False,"noerror":True,"message": None}
            logging.info(f"Running {n._name}")   
            n.run_async(self.done_task_callback,self.error_task_callback)

        logging.info(f"{self._analysis_id} completed")

    def done_task_callback(self,task):
        logging.info(f"Received done callback from {task}")
        
        self._tasks[task]["done"] = True
        done_all = all([a["done"] for _,a in self._tasks.items()])
        logging.info(f"Closed all tasks? {done_all}")
        if done_all:
            aid = self._analysis_id
            
            if all([a["noerror"] for _,a in self._tasks.items()]):
                post_new_status(aid,"Completed","workflow completed") 
                logging.info("Done all peacefully! closing analysis")
            else:
                message = "\n".join([a["message"] for _,a in self._tasks.items() if a["message"]])
                post_new_status(aid,"Error",message) 
                logging.info("Done all with error! closing analysis")
            self._busy = False
    
    def error_task_callback(self,task,error):
        logging.error(error)
        self._tasks[task]["noerror"] = False
        # TODO: handle this differently
        self.done_task_callback(task)

    def status(self):
        return self._tasks

class WorkflowFactory:
    def __init__(self):
        pass

    @staticmethod
    def create_from_dict(workflow_structure,analysis):    
        # Getting nodes dictionary
        nodes_json = workflow_structure["nodes"]
        # Creating dict to escape for loop in fill args
        workflow_args = {n["name"]: n for n in workflow_structure["feeds"]["params"]}
        # replace workflow placeholders with user inputs
        fill_args(nodes_json,workflow_args,analysis["args"])

        # This is important for output nodes
        nodes = {}
        for n in nodes_json:
            logging.info(f"creating node {n}")
            if "nodes" in n:
                n["nodes"] = [nodes[i] for i in n["nodes"]]
            elif "node" in n:
                n["node"] = nodes[n["node"]]
            nodes[n["name"]] = create_node_from_dict(n)
        
        
        logging.info(nodes)

        edgelist = workflow_structure["edges"]
        for f,t in edgelist:
            nodes[f].add_sink(nodes[t])
        
        starterslist = workflow_structure["starters"]
        starters = [nodes[s] for s in starterslist]
    
        run_order = workflow_structure["run_order"]
        nodes_in_order = [nodes[n] for n in run_order]
        
        w = Workflow(analysis["id"],nodes,starters,nodes_in_order)

        return w