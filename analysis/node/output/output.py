from ..node import Node

class Output(Node):
    def __init__(self, name,prefix):
        Node.__init__(self,name)
        self._prefix = prefix

    def write_meta(self):
        raise NotImplementedError

