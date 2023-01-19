from .agg import *
from .ai import *
from .filter import *
from .output import *
from .source import *
from .controller import *
from .draw import *
from .edit import *
from .msc import *



def create_node_from_dict(node_dict,**args):
    node_type = node_dict.pop("type")
    node_class = globals()[node_type]
    return node_class(**node_dict)