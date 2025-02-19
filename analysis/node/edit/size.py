import logging
import numpy as np
from PIL import Image
from ..node import Node

class Resizer(Node):
    def __init__(self, name,size):
        Node.__init__(self,name)
        self._enabled = True
        if size == "-1":
            self._enabled = False
        sp = size.split(",")
        self._width = int(sp[0])
        self._height = int(sp[1])

    def run(self):
        # expecting items of type FalcoeyeDetction
        #logging.info(f"Running {self._name} with resolution {self._width}X{self._height}")
        while self.more():
            item = self.get()
            if self._enabled:
                # assuming FalcoeyeFrame
                #logging.info(f"Resizing frame to {self._width}X{self._height}")
                item.resize(self._width,self._height)
                logging.info(f"Frame resized {item.size[1]}X{item.size[0]}")
            self.sink(item)
    
    def close(self):
        self.close_sinks()

class TritonPreprocessor(Node):
    def __init__(self, name, input_shape, letter_box=True):
        Node.__init__(self, name)
        
        # Handle both [height, width] and [batch, channels, height, width] formats
        if len(input_shape) == 2:
            self._height, self._width = input_shape
            self._input_shape = [1, 3, self._height, self._width]  # Add batch and channel dims
        elif len(input_shape) == 4:
            self._input_shape = input_shape
            self._height = input_shape[2]
            self._width = input_shape[3]
        else:
            raise ValueError("input_shape must be either [height, width] or [batch, channels, height, width]")
            
        self._letter_box = letter_box
        
    def preprocess_frame(self, img):
        """Preprocess a single frame for Triton inference"""
        # Convert numpy array to PIL Image if necessary
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
            
        img_w, img_h = img.size
        
        if self._letter_box:
            new_h, new_w = self._height, self._width
            offset_h, offset_w = 0, 0
            
            # Calculate new dimensions preserving aspect ratio
            if (new_w / img_w) <= (new_h / img_h):
                new_h = int(img_h * new_w / img_w)
                offset_h = (self._height - new_h) // 2
            else:
                new_w = int(img_w * new_h / img_h)
                offset_w = (self._width - new_w) // 2
            
            # Resize image
            resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            
            # Create new image with padding
            new_img = Image.new('RGB', (self._width, self._height), (127, 127, 127))
            new_img.paste(resized, (offset_w, offset_h))
            img = new_img
        else:
            img = img.resize((self._width, self._height), Image.Resampling.BILINEAR)

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Create RGB version for FalcoeyeFrame
        rgb_array = np.array(img, dtype=np.uint8)
            
        # Create preprocessed tensor for inference
        img_tensor = np.array(img, dtype=np.float32)
        img_tensor = img_tensor.transpose((2, 0, 1))  # HWC to CHW format
        img_tensor /= 255.0
        
        # Verify tensor format
        assert img_tensor.dtype == np.float32
        return rgb_array, img_tensor

    def run(self):
        """Process frames in the workflow"""
        logging.info(f"Running {self._name} with input shape {self._input_shape}")
        
        while self.more():
            item = self.get()
            try:
                # Store original dimensions for postprocessing
                if isinstance(item.frame, np.ndarray):
                    item.original_dims = item.frame.shape[:2]  # (height, width)
                else:
                    item.original_dims = item.frame.size[::-1]  # PIL size is (width, height)
                
                # Process the frame
                rgb_array, tensor_data = self.preprocess_frame(item.frame)
                
                # Update the frame with resized RGB data
                item.set_frame(rgb_array)
                
                # Store preprocessed tensor data
                item.tensor = tensor_data
                
                logging.info(f"Frame processed to shape {tensor_data.shape}")
                self.sink(item)
                
            except Exception as e:
                logging.error(f"Error preprocessing frame: {str(e)}")
                continue

    def close(self):
        self.close_sinks()