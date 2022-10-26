from threading import Thread
import cv2
import numpy as np

from .source import Source,FalcoeyeFrame
import logging
from ...utils import download_file, rm_file

class VideoFileSource(Source):
    def __init__(self, name, filename,sample_every,length=-1,**kwargs):
        Source.__init__(self,name)
        self._filename = filename
        self._sample_every = int(sample_every)
        self._num_frames = int(length)
        self._frames_per_second = -1
        self._reader = None
        self.width = -1
        self.height = -1

    def open(self):
        # Downloading in the /temp from cloud storage
        self._alter_filename = download_file(self._filename)
        self._reader = cv2.VideoCapture(self._alter_filename)
        self.width = int(self._reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frames_per_second = self._reader.get(cv2.CAP_PROP_FPS)
        self._length = int(self._reader.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info(f"opened file length: {self._length}, fps: {self._frames_per_second} width: {self.width} height: {self.height}")
        if self._num_frames < 0:
            self._num_frames = self._length

    def seek(self,n):
        self._reader.set(cv2.CAP_PROP_POS_FRAMES,n)

    def run(self):
        self.open()
        counter = 0
        count = 0
        logging.info(f"Start streaming from {self._filename}")
        while counter < self._num_frames:
            hasFrame, frame = self._reader.read()
            if not hasFrame:
                logging.info("No more frames. Breaking!")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                  
            logging.info(f"Frame {counter}/{self._num_frames}")
            self.sink(FalcoeyeFrame(frame,count,counter,"frame"))
            #logging.info(f"Frame {counter}/{self._num_frames} sinked")
            count += 1
            counter += self._sample_every
            if counter > self._length:
                break
            self.seek(counter)

        logging.info(f"Streaming completed")
        if self._done_callback:
            self._done_callback(self._name)
        
        # deleting temp file from cloud compute (will be ignored on local)
        #rm_file(self._alter_filename)
        
        self.close()
        self.close_sinks()

    def run_async(self,done_callback,error_callback):
        self._done_callback = done_callback
        self._error_callback = error_callback
        self._thread = Thread(target=self.run,daemon=True)
        self._thread.start()

    def close(self):
        self._reader.release()
    

