from ..node import Node
import logging

def count(item,si):
    item.add_meta(si,item.count)

def diameter(item,si):
    diameter = []
    for i in range(item.count):
        x1,y1,x2,y2 = item.get_box(i)
        diameter.append(round(((x2-x1)**2 + (y2-y1)**2)**0.5,3))
    item.add_meta(si,diameter)

def normalized_diameter(item,si):
    diameter = []
    height,width,_ = item.size
    item_diameter = (width**2+height**2)**0.5
    for i in range(item.count):
        x1,y1,x2,y2 = item.get_box(i)
        idiam = ((x2-x1)**2 + (y2-y1)**2)**0.5
        idiam /= item_diameter
        diameter.append(round(idiam,3))
    item.add_meta(si,diameter)

FUNCTIONS = {
    "count": count,
    "diameter": diameter,
    "normalized_diameter": normalized_diameter
}

class BBoxInfo(Node):
    def __init__(self, name, operations,store_in):
        Node.__init__(self,name)
        self._operations = [FUNCTIONS[o] for o in operations]
        self._store_in = store_in
    
    def run(self):
        while self.more():
            ai_item = self.get()
            for op,si in zip(self._operations,self._store_in):
                op(ai_item,si)
            self.sink(ai_item)

