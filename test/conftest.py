import pytest
import sys
sys.path.insert(0,"../")
import json
import logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
from ..analysis.utils import get_service
import requests


@pytest.fixture
def backend():
    URL = get_service("falcoeye-backend")
    logging.info(f"Will use {URL} as backend")
    return URL

@pytest.fixture
def register(backend):
    info={
        "pass": [
            "email or username already exists",
            "user registered"
        ],
        "args":{
            "email": "falcoeye-test@falcoeye.io",
            "username": "falcoeye_test",
            "name": "falcoeye test",
            "password": "falcoeye-test"
        }
    }
    logging.info(f"Registering {info['args']['email']}")
    res = requests.post(f"{backend}/auth/register", json=info['args'])
    resdict = res.json()
    message = resdict["message"]
    if message in info["pass"]:
        logging.info(f"Registration passed with message: {message}")
        return True
    else:
        logging.error(f"Registration failed with message: {message}")
        return False

@pytest.fixture
def login(backend):
    info={
        "pass": [
            "successfully logged in"
        ],
        "args":{
            "email": "falcoeye-test@falcoeye.io",
            "password": "falcoeye-test"
        }
    }
    args = info["args"]
    logging.info(f"Logging in with {args['email']}")
    res = requests.post(f"{backend}/auth/login", json=info['args'])
    resdict = res.json()
    message = resdict["message"]
    if message in info["pass"]:
        logging.info(f"Logging in passed with message: {message}")
        return resdict
    else:
        logging.error(f"Logging in failed with message: {message}")
        return False

@pytest.fixture
def test_user_token(backend,register,login):
    assert login
    return f'JWT {login["access_token"]}'

@pytest.fixture
def workflows(backend,test_user_token):
    header = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "X-API-KEY": test_user_token
    }
    resp = requests.get(f"{backend}/api/workflow/?page=1&per_page=10&inline=true", headers=header)
    logging.info(f"Workflow request status: {resp.status_code}")
    resdict = resp.json()
    message = resdict["message"]
    assert "workflow" in resdict
    workflows = resdict["workflow"]
    assert message == "workflow data sent"
    assert len(workflows) > 0
    return workflows

@pytest.fixture
def videos(backend,test_user_token):
    header = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "X-API-KEY": test_user_token
    }
    resp = requests.get(f"{backend}/api/media/?page=1&per_page=10",headers=header)
    resdict = resp.json()
    message = resdict["message"]
    assert message == "media data sent"
    assert "media" in resdict
    logging.info(resdict)
    assert len(resdict["media"]) > 0
    return resdict["media"]

@pytest.fixture
def analysis(backend,test_user_token):
    header = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "X-API-KEY": test_user_token
    }
    resp = requests.get(f"{backend}/api/analysis/?page=1&per_page=10&inline=true", headers=header)
    logging.info(f"Analysis request status: {resp.status_code}")
    resdict = resp.json()
    message = resdict["message"]
    assert "analysis" in resdict
    analysis = resdict["analysis"]
    assert message == "analysis data sent"
    return analysis
