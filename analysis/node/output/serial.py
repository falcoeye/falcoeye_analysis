from ..output import Output
import os
import json
import logging
from ...utils import mkdir,exists
from datetime import datetime
import io
from PIL import Image
import glob

class Serializer(Output):
    def __init__(self, name,prefix):
        Output.__init__(self,name,prefix)
        self._prefix = prefix
        if self._prefix[-1] == "/":
            self._prefix = self._prefix[:-1]
        self._meta = {
            "type": "serialized",
            "filenames": []
        }
        logging.info(f"Creating folder {prefix}")
        mkdir(prefix)

    def write_meta(self):
        self._meta["filenames"] = [os.path.basename(f) for f in glob.glob(f"{self._prefix}/*.json")]
        metafile = f"{self._prefix}/meta.json" 
        logging.info(f"Creating meta data in {metafile}")
        with open(
                os.path.relpath(metafile), "w"
            ) as f:
                f.write(json.dumps(self._meta))

    def run(self):
        """
        Safe node: input is assumed to be valid or can be handled properly
        """
        # TODO: refactor
        try:
            while self.more():
                item = self.get()
                # can only serizlize FalcoeyeFrame or its wrappers
                filename = f"{self._prefix}/{item.framestamp}.json"
                with open(filename,"w") as f:
                    f.write(json.dumps(item.to_dict(),indent=4))
            self.write_meta()
        except Exception as e:
            logging.error(e)
    