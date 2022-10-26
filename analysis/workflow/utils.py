import logging 

def fill_value(k,v,n,workflow_args,args):
    if type(v) == str and v[0] == "$" and v[1:] in args:
        n[k] = args[v[1:]]
    elif  type(v) == str and v[0] == "$" and v[1:] in workflow_args:
        if "default" in workflow_args[v[1:]]:
            n[k] = workflow_args[v[1:]]["default"]
    elif type(v) == dict:
        for k2,v2 in v.items():
            fill_value(k2,v2,v,workflow_args,args)

def fill_args(structure,workflow_args,args):
    for n in structure:
        for k,v in n.items():
            logging.info(f"{k}: {v}")
            fill_value(k,v,n,workflow_args,args)
            logging.info(f"{k}:{n[k]}")
