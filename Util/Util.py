from Util.AudioUtil import AudioUtil
from HParams import HParams
class Util:
    def __init__(self,h_params:HParams):
        self.h_params = h_params
        self.audio = AudioUtil(h_params)
