from ..node import Node
import pytesseract
import logging

class FalcoeyeOCRNode(Node):
	def __init__(self, name, 
	ocr_slice=None,
	store_in="ocr"
	):
		Node.__init__(self,name)
		self._ocr_slice = ocr_slice
		self._store_in = store_in
		if type(ocr_slice) == str:
			slc = ocr_slice.split(",")
			self._x1,self._y1,self._x2,self._y2 = [int(i) for i in slc]
		else:
			self._x1,self._y1,self._x2,self._y2 = 0,-1,0,-1
	
	def run(self):
		"""
		Safe node: the input is assumed to be valid or can be handled properly, no need to catch
		"""
		logging.info(f"Running falcoeye ocr node")
		while self.more():
			ai_item = self.get()
			frame = ai_item.frame
			value = pytesseract.image_to_string(
				frame[self._y1:self._y2,self._x1:self._x2])
			# TODO: what if not FalcoeyeAIWrapper
			ai_item.add_meta(self._store_in,value)
			self.sink(ai_item)

	