from ..node import Node
import pandas as pd
import logging

class PointCounter(Node):
    def __init__(self, name):
        Node.__init__(self,name)

    def run(self):
        table = []
        logging.info(f"Running {self.name}")
        while self.more():
            logging.info(f"New item for {self.name}")
            item = self.get()
            row = [item.timestamp,item.framestamp,item.count]
            table.append(row)
        df = pd.DataFrame(table,columns=["Timestamp","Frame_Order","Count"])
        logging.info(f"\n{df}")
        self.sink(df)