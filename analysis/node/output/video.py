from ..output import Output
import os
import cv2
import json
import logging
from ...utils import mkdir, put, random_string,tempdir,rm_file,exists
from datetime import datetime
import io
from PIL import Image

class VideoWriter(Output):
    def __init__(self, name,prefix,frames_per_second=30):
        Output.__init__(self,name,prefix)
        self._prefix = prefix
        if self._prefix[-1] == "/":
            self._prefix = self._prefix[:-1]
        self._frames_per_second = frames_per_second
        self._writer = None
        self._meta = {
            "type": "media",
            "filenames": []
        }
        logging.info(f"Creating folder {prefix}")
        mkdir(prefix)

    def write_meta(self):
        metafile = f"{self._prefix}/{self._name}_meta.json" 
        logging.info(f"Creating meta data in {metafile}")
        with open(
                os.path.relpath(metafile), "w"
            ) as f:
                f.write(json.dumps(self._meta))
        
    def open_writer(self, filename,width, height):
        
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        self._writer = cv2.VideoWriter(
            filename,
            fourcc=fourcc,
            fps = 30,
            frameSize=(width, height),
            isColor=True,
        )
        logging.info(f"Writer for video {filename} opened")

    def close_writer(self):
        if self._writer:
            self._writer.release()

    def run(self):
        """
        Safe node: input is assumed to be valid or can be handled properly
        """

        # TODO: refactor
        try:
            while self.more():
                logging.info(f"New video to write for {self._name}")
                count = 0
                item = self.get()
                self._meta["filenames"] = []

                filename = f"{self._prefix}/{self._name}_{count}.mp4"
                thumbfile = f'{self._prefix}/{self._name}_{count}_260.jpg'
                self._meta["filenames"].append(f'{self._name}_{count}.mp4')
                while exists(filename):
                    count += 1
                    filename = f'{self._prefix}/{self._name}_{count}.mp4'
                    thumbfile = f'{self._prefix}/{self._name}_{count}_260.jpg'
                    self._meta["filenames"].append(f'{self._name}_{count}.mp4')
                
                logging.info(f"New video to write to {filename}")
            
                height,width,_ = item[0].size
                rstring = random_string()
                tempfile =  f'{tempdir()}/{datetime.now().strftime("%m_%d_%Y")}_{rstring}.mp4'
                
                self.open_writer(tempfile,width,height)
                
                sorted_frames = sorted(item, key=lambda x: x.framestamp, reverse=False)
                for det in sorted_frames:
                    self._writer.write(det.frame_bgr)

                
                logging.info(f"Creating thumbnail {thumbfile}")
                img = Image.fromarray(sorted_frames[0].frame)
                with open(os.path.relpath(thumbfile), "wb") as f:
                    byteImgIO = io.BytesIO()
                    img.thumbnail((260,260))
                    logging.info(f"thumbnail size {img.size}")
                    img.save(byteImgIO, "JPEG")
                    byteImgIO.seek(0)
                    byteImg = byteImgIO.read()
                    f.write(byteImg)
                
                self.close_writer()
                
                logging.info(f"Putting {tempfile} into {filename}")
                put(tempfile,filename)
                rm_file(tempfile)

            self.write_meta()
        except Exception as e:
            logging.error(e)
    

class ActiveVideoWriter(VideoWriter):
    def __init__(self,name,prefix,frames_per_second=30):
        VideoWriter.__init__(self,name,prefix,frames_per_second)
        count = 0
        self._filename = f"{self._prefix}/{self._name}_{count}.mp4"
        self._thumbfile = f'{self._prefix}/{self._name}_{count}_260.jpg'
        self._meta["filenames"] = []
        self._meta["filenames"].append(f'{self._name}_{count}.mp4')
        
        while exists(self._filename):
            count += 1
            self._filename = f'{self._prefix}/{self._name}_{count}.mp4'
            self._thumbfile = f'{self._prefix}/{self._name}_{count}_260.jpg'
            self._meta["filenames"].append(f'{self._name}_{count}.mp4')
        self.write_meta()
        rstring = random_string()
        self._tempfile =  f'{tempdir()}/{datetime.now().strftime("%m_%d_%Y")}_{rstring}.mp4'            
        
    def run(self):
        try:
            while self.more():
                item = self.get()
                if self._writer is None:  
                    logging.info("First item to video writer")
                    logging.info(f"{type(item)} {item.size}")
                    height,width,_ = item.size
                    logging.info(f"Opening a new video with resolution {width}X{height}")
                    self.open_writer(self._tempfile,width,height)
                    logging.info(f"Creating thumbnail {self._thumbfile}")
                    img = Image.fromarray(item.frame)
                    with open(os.path.relpath(self._thumbfile), "wb") as f:
                        byteImgIO = io.BytesIO()
                        img.thumbnail((260,260))
                        logging.info(f"thumbnail size {img.size}")
                        img.save(byteImgIO, "JPEG")
                        byteImgIO.seek(0)
                        byteImg = byteImgIO.read()
                        f.write(byteImg)
                logging.info(f"item is of type {type(item)}")
                self._writer.write(item.frame_bgr)
        except Exception as e:
            logging.error(e)
    
    def close(self):
        self.close_writer()
        logging.info(f"Putting {self._tempfile} into {self._filename}")
        put(self._tempfile,self._filename)
                