
from ..output import Output
import os
import logging
import json
from ...utils import rmtree,mkdir,rm_file, exists
import pandas as pd

def append_row(df, row):
    return pd.concat([
                df, 
                pd.DataFrame([row], columns=row.index)]
           ).reset_index(drop=True)

class CSVWriter(Output):
    def __init__(self, name,prefix,
    xaxis = "",
    yaxis = "",
    overwrite=True):
        Output.__init__(self,name,prefix)
        self._filename = f"{prefix}/{name}.csv"
        self._xaxis = xaxis
        self._yaxis = yaxis
        self._df = None
        if exists(os.path.relpath(self._filename)):
            rm_file(self._filename)
        
        logging.info(f"Creating folder {prefix}")
        mkdir(prefix)
              
    def write_meta(self):
        meta = {
            "type": "csv",
            "filename": f"{os.path.basename(self._filename)}",
            "x-axis": self._xaxis,
            "y-axis": self._yaxis
        }
        metafile = f"{self._prefix}/{self._name}_meta.json"
        logging.info(f"Creating meta data in {metafile}")
        with open(
                os.path.relpath(metafile), "w"
            ) as f:
                f.write(json.dumps(meta))

    def run(self):
        """
        Safe node: input is assumed to be valid or can be handled properly
        """
        # TODO: refactor
        logging.info(f"Running {self.name} {self._filename}")
        try:
            self.write_meta()
            while self.more():
                item = self.get()
                logging.info(f"New item for {self.name}\n{item}")
                if self._df is None:
                    self._df = item
                else:
                    self._df = pd.concat([self._df,item])

                with open(
                    os.path.relpath(self._filename), "w") as f:
                    logging.info(f"Writing to {os.path.relpath(self._filename)}")
                    f.write(self._df.to_csv(None,index=False))   
        except Exception as e:
            logging.erro(e)

class TableWriter(Output):
    def __init__(self, name,prefix,columns,overwrite=True):
        Output.__init__(self,name,prefix)
        self._filename = f"{prefix}/{name}.csv"
        self._columns = columns
        self._df = pd.DataFrame(columns=["Timestamp","Frame_Order"]+self._columns)
        if exists(os.path.relpath(self._filename)):
            rm_file(self._filename)
        
        logging.info(f"Creating folder {prefix}")
        mkdir(prefix)

    def write_meta(self):
        meta = {
            "type": "table",
            "filename": f"{os.path.basename(self._filename)}",
            "columns": self._columns
        }
        metafile = f"{self._prefix}/{self._name}_meta.json"
        logging.info(f"Creating meta data in {metafile}")
        with open(
                os.path.relpath(metafile), "w"
            ) as f:
                f.write(json.dumps(meta))

    def run(self):
        """
        Safe node: input is assumed to be valid or can be handled properly
        """
        # TODO: refactor
        logging.info(f"Running {self.name} {self._filename}")
        try:
            self.write_meta()
            while self.more():
                item = self.get()
                logging.info(f"New item for {self.name}\n{item}")
                  
                row = {'Timestamp':item.timestamp, 'Frame_Order':item.framestamp}
                for c in self._columns:
                    row[c] = item.get_meta(c)
                    
                new_row = pd.DataFrame(row)
                self._df = pd.concat([self._df, new_row], ignore_index=True)
                with open(
                    os.path.relpath(self._filename), "w") as f:
                    logging.info(f"Writing to {os.path.relpath(self._filename)}")
                    f.write(self._df.to_csv(None,index=False))
                    
        except Exception as e:
            logging.erro(e) 