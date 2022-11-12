
from ..node import Node
import logging
from ..ai.segmentation import FalcoeyeSegmentation
from ..source.source import FalcoeyeFrame
from PIL import Image


class SegmentationBlender(Node):
    def __init__(self, name,alpha=0.5):
        Node.__init__(self,name)
        self._current_segmentation = None
        self._alpha = alpha

    def run(self):
        # expecting items of type FalcoeyeDetction
        while self.more():
            item = self.get()
            if self._current_segmentation is None and type(item) != FalcoeyeSegmentation:
                frame = item
            elif type(item) == FalcoeyeSegmentation:
                logging.info(f"Running {self._name} on item {item.framestamp} with new segmentation")
                self._current_segmentation = item
                #blended image numpy array
                blended = self._current_segmentation.blend(alpha=self._alpha,inplace=False)
                # set the internal FalcoeyeFrame frame of the segmentation to the blended
                item.set_frame(blended)
                frame = item
            elif type(item) == FalcoeyeFrame:
                logging.info(f"Running {self._name} on item {item.framestamp}")
                frame = item.blend(self._current_segmentation.rgb_map,alpha=self._alpha,inplace=True)
            else:
                raise NotImplementedError
            self.sink(frame)

                