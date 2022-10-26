
from ..node import Node
from .stream import AngelCamSource,YoutubeSource,RTSPSource,M3U8Source
from .video import VideoFileSource
import logging
def determine_source(**kwargs):
    logging.info(f"Determining source {kwargs}")
    url = kwargs.get("url", None)
    if url:
        if "youtube" in url:
            return YoutubeSource
        elif "angelcam" in url:
            return AngelCamSource
        elif url.endswith(".m3u8"):
            return M3U8Source
    filename = kwargs.get("filename", None) 
    if filename:
        return VideoFileSource
    
    return RTSPSource
    

class DynamicSource(Node):
    def __new__(cls,*args,**kwargs): 
        logging.info(f"Creating dynamic source with args {kwargs}")
        node_class = determine_source(**kwargs)
        return node_class(**kwargs)
    