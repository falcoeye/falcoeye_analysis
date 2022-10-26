import os
import requests
import logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
import json
from analysis.utils import get_service
from analysis.workflow.workflow import WorkflowFactory


URL = get_service("falcoeye-backend")

workflow_user = os.getenv("WORKFLOW_USER")
workflow_password = os.getenv("WORKFLOW_PASSWORD")

payload =  {
        "email": workflow_user.strip(),
        "password": workflow_password.strip()
}
logging.info(f"Logging in {URL}")
r = requests.post(f"{URL}/auth/login", json=payload)

assert "access_token" in r.json()
access_token = r.json()["access_token"]
os.environ["JWT_KEY"] = f'JWT {access_token}'


analysis_file = os.getenv("ANALYSIS_PATH")
with open(analysis_file) as f:
    data = json.load(f)

workflow_struct = data["workflow"]
analysis = data["analysis"]
logging.info(f"Analysis id: {analysis['id']}")
logging.info(data)


# Creating workflow handler
workflow = WorkflowFactory.create_from_dict(workflow_struct,analysis)
workflow.run_sequentially_async()

while workflow._busy:
    pass