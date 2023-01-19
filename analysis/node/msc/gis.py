from ..node import Node
import logging

class FalcoeyeGPSInterpreterNode(Node):
	def __init__(self, name, get_from):
		Node.__init__(self,name)
		self._get_from = get_from
	
	def _clean(self,gps_value):
		gps_value = gps_value.strip()
		gps_value = gps_value.replace("°"," ").replace("'","").replace(". ",".")
		gps_value = gps_value.replace(" N","N").replace(" E","E").replace("£","E")
		return gps_value
	
	def _deg_to_dec(self,deg,letter):
		d,m = deg.split()
		dc = float(d) + float(m.strip(letter))/60
		return dc

	def run(self):
		while self.more():
			ai_item = self.get()
			gps_value = ai_item.get_meta(self._get_from)
			# clean
			gps_value = self._clean(gps_value) 
			try:	
				east,north = gps_value.split(",")
				north = round(self._deg_to_dec(north.strip(),"N"),3)
				east = round(self._deg_to_dec(east.strip(),"E"),3)
				logging.info(f"Parsed GIS value {north}, {east}")
			except:
				logging.warning(f"Couldn't parse GPS Value: {gps_value}")
				north,east = None,None
			
			ai_item.add_meta("latitude",north)
			ai_item.add_meta("longitude",east)
			self.sink(ai_item)