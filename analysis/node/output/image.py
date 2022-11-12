from ..output import Output
import os
import cv2
import json
import logging
from ...utils import mkdir, put, random_string,tempdir,rm_file,exists
from datetime import datetime
import io
from PIL import Image
import glob

class ImageWriter(Output):
    def __init__(self, name,prefix):
        Output.__init__(self,name,prefix)
        self._prefix = prefix
        if self._prefix[-1] == "/":
            self._prefix = self._prefix[:-1]
        self._meta = {
            "type": "media",
            "filenames": []
        }
        logging.info(f"Creating folder {prefix}")
        mkdir(prefix)

    def write_meta(self):
        self._meta["filenames"] = [os.path.basename(f) for f in glob.glob(f"{self._prefix}/*.jpg")]
        metafile = f"{self._prefix}/{self._name}_meta.json" 
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
                filename = f"{self._prefix}/{item.framestamp}.jpg"
                
                img = Image.fromarray(item.frame)
                logging.info(f"writing image for {item.framestamp}")
                img.save(filename)
                #thumbfile = f'{self._prefix}/{item.framestamp}_260.jpg'
                # logging.info(f"Creating thumbnail {thumbfile}")
                # with open(os.path.relpath(thumbfile), "wb") as f:
                #     byteImgIO = io.BytesIO()
                #     img.thumbnail((260,260))
                #     logging.info(f"thumbnail size {img.size}")
                #     img.save(byteImgIO, "JPEG")
                #     byteImgIO.seek(0)
                #     byteImg = byteImgIO.read()
                #     f.write(byteImg)
                
                # self.close_writer()
            self.write_meta()
        except Exception as e:
            logging.error(e)
    