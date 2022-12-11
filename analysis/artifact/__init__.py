

import logging
import os
def get_model_server(model_name, 
    vendor,model_version,
    protocol,run_if_down=True,
    input_name="input_tensor"):
    logging.info(f"Getting model server for {model_name} with input name {input_name}")
    deployment = os.getenv("DEPLOYMENT","local")
    # if deployment == "local":
    #     from .local import get_service_server as gsa
    #     return gsa(model_name,model_version,protocol,run_if_down=run_if_down)
    # elif deployment == "k8s":
    from .k8s import get_service_server as gsa
    return gsa(model_name,vendor,
        model_version,protocol,
        run_if_down=run_if_down,
        input_name=input_name)