import re
import subprocess as sp
from queue import Queue
from threading import Thread
import cv2
import numpy as np
import requests
import streamlink
import logging
import time
import os

from .source import Source, FalcoeyeFrame

class StreamingSource(Source):
    def __init__(self, name, sample_every=5, length=30,**kwargs):
        Source.__init__(self,name)
        self._running = False
        self._sample_every = int(sample_every)
        self._length = float(length)
        self._streamer = None
        self._trial = 10
    
    def open(self):
        self._running = True
    
    def close(self):
        self._running = False
    
    def read(self):
        raise NotImplementedError

    def run(self):
        self.open()
        try:
            count = 0
            t_end = time.time() + self._length
            c_time = time.time()
            logging.info(f"Entering stream loop {c_time}")

            while  c_time < t_end:
                logging.info(f"Reading new frame")
                # fetching new frame
                count = 0
                hasFrame, frame = self.read()
                while not hasFrame and count < self._trial:
                    hasFrame, frame = self.read()
                    count += 1

                logging.info(f"New frame read? {hasFrame} {count} {c_time}")
                if not hasFrame and count == self._trial:
                    logging.error("Stream is not reachable")
                    break
                elif not hasFrame:
                    logging.error("Unknown stream issue")
                    break
            
                logging.info(f"New frame fetched {count} {c_time}")
                # sinking data
                self.sink(FalcoeyeFrame(frame,count,c_time,"epoch"))
                # sleeping for $sample_every seconds
                count += 1
                time.sleep(self._sample_every)
                c_time = time.time()

            logging.info(f"Exiting stream loop {c_time}")
            if self._done_callback:
                self._done_callback(self._name)
        except Exception as e:
            logging.error(str(e))
            self._error_callback(self._name,"Error in streaming")
        
        self.close()
        self.close_sinks()
    
    def run_async(self,done_callback,error_callback):
        self._done_callback = done_callback
        self._error_callback = error_callback
        self.open()
        self._thread = Thread(target=self.run, args=(),daemon=True)
        self._thread.start()

class StreamingServerSource(StreamingSource):
    def __init__(self, name, url,  resolution="best", sample_every=5, length=30,**kwargs):
        StreamingSource.__init__(self,name,sample_every,length)
        self._url = url
        self._resolution = resolution
        self._width = -1
        self._height = -1
        self._frames_per_second = 30
 
    @staticmethod
    def create_stream_pipe(url, resolution):
        if url == None:
            return None

        try:
            streams = streamlink.streams(url)
        except streamlink.exceptions.NoPluginError:
            logging.warning(f"Warning: NO STREAM AVAILABLE in {url}")
            return None
        
        logging.info(f"Starting streaming with resolution {resolution}")
        ffmpeg = "/usr/local/bin/ffmpeg"
        if not os.path.exists(ffmpeg):
            ffmpeg = "/usr/bin/ffmpeg"

        stream = streams[resolution]
        pipe = sp.Popen(
            [
                ffmpeg,
                "-i",
                stream.url,
                "-loglevel",
                "quiet",  # no text output
                "-an",  # disable audio
                "-f",
                "image2pipe",
                "-pix_fmt",
                "bgr24",
                "-vcodec",
                "rawvideo",
                "-",
            ],
            stdin=sp.PIPE,
            stdout=sp.PIPE,
        )
        logging.info(f"Pipe instantiated")
        return pipe

    def open(self):
        StreamingSource.open(self)

    def close(self):
        logging.info("Closing streaming server")
        StreamingSource.close(self)
        self._streamer.kill()
        logging.info("Streaming server closed")
    
    def read(self):
        try:
            logging.info("Fetching new frame from stream")
            raw_image = self._streamer.stdout.read(
                        self._height * self._width * 3
                    )  # read length*width*3 bytes (= 1 frame)
            logging.info("Frame fetched. Creating numpy")
            frame = np.fromstring(raw_image, dtype="uint8").reshape(
                (self._height, self._width, 3)
            )
            logging.info("Frame fetching succeeded")
            return True,frame
        except Exception as error:
            logging.error("Couldn't fetch frame")
            return False,None
        
class AngelCamSource(StreamingServerSource):
    resolutions = {"best": {"width": 1920, "height": 1080}}

    def __init__(self, url,  resolution="best", sample_every=5, length=60,**kwargs):
        StreamingServerSource.__init__(self, name, url, resolution, sample_every, length)
        self._m3u8 = None
        self._width = AngelCamSource.resolutions[resolution]["width"]
        self._height = AngelCamSource.resolutions[resolution]["width"]

    def open(self):
        StreamingServerSource.open(self)
        c = requests.get(self.self._url).content.decode("utf-8")
        self._m3u8 = re.findall(r"\'https://.*angelcam.*token=.*\'", c)[0].strip("'")
        self._streamer = StreamingServerSource.create_pipe(self._m3u8,self._resolution)

class YoutubeSource(StreamingServerSource):
    resolutions = {
        "240p": {"width": 426, "height": 240},
        "360p": {"width": 640, "height": 360},
        "480p": {"width": 854, "height": 480},
        "720p": {"width": 1280, "height": 720},
        "1080p": {"width": 1920, "height": 1080},
    }

    def __init__(
        self, name,url, resolution="1080p", sample_every=5, length=30,**kwargs):
        StreamingServerSource.__init__(self, name, url, resolution, sample_every, length)
        self._width = YoutubeSource.resolutions[resolution]["width"]
        self._height = YoutubeSource.resolutions[resolution]["height"]
        self._streamer = None

    def open(self):
        StreamingServerSource.open(self)
        logging.info(f"Creating pipe with {self._url}, {self._resolution}")
        self._streamer = StreamingServerSource.create_stream_pipe(self._url, self._resolution)
    
class RTSPSource(StreamingSource):
    def __init__(self,name,host,port=554,username=None,password=None,sample_every=5, length=60,**kwargs):
        StreamingSource.__init__(self,name,sample_every,length)
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._url = f"rtsp://"
        if username:
            self._url += f'{username}:{password}@'
        self._url += f'{host}:{port}/Streaming/Channels/1'
        self._running = False

    def open(self):
        StreamingSource.open(self)
        self._streamer = cv2.VideoCapture(self._url)

    def read(self):
        ret, frame = self._streamer.read()
        return ret,frame

    def close(self):
        StreamingSource.close(self)
        if self._streamer:
            self._streamer.release()
        self._streamer = None
 
class M3U8Source(StreamingSource):
    resolutions = {"best": {"width": 320, "height": 180}}
    def __init__(self,name,url,sample_every=5, length=60,**kwargs):
        StreamingSource.__init__(self,name,sample_every,length)
        self._url = url

    def open(self):
        StreamingSource.open(self)
        self._streamer = cv2.VideoCapture(self._url)

    def read(self):
        ret, frame = self._streamer.read()
        return ret,frame

    def close(self):
        StreamingSource.close(self)
        if self._streamer:
            self._streamer.release()
        self._streamer = None


  
