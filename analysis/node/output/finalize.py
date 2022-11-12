from ..output import Output
import os
import cv2
import json
from datetime import datetime
import io
from PIL import Image
import glob

class Finalizer(Output):
    def __init__(self, name,prefix):
        Output.__init__(self,name,prefix)
        self._prefix = prefix
        if self._prefix[-1] == "/":
            self._prefix = self._prefix[:-1]

    def run(self):
        """
        Safe node: input is assumed to be valid or can be handled properly
        """
        # TODO: refactor
        try:
            meta_files = glob.glob(f"{self._prefix}/*_meta.json")
            meta_names = [os.path.basename(f).replace("_meta.json","") for f in meta_files]
            
            # For current front-end version we need to do this
            if len(meta_names) == 1:
                with open(meta_files[0]) as f:
                    metas = json.load(f)
            else:
                metas = {}
                for f,n in zip(meta_files,meta_names):
                    with open(f) as f:
                        metas[n] = json.load(f)

            metafile = f"{self._prefix}/meta.json" 
            with open(metafile, "w") as f:
                f.write(json.dumps(metas))

        except Exception as e:
            logging.error(e)