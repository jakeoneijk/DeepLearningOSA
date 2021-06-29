import torch
from torchsummary import summary
from HParams import HParams

class TestModelIO():
    def __init__(self,model,h_params:HParams):
        self.test_model = model
        channel = None
        time_dimenstion = None
        self.test_model_input = (channel,time_dimenstion)
    
    def test(self):
        summary(self.test_model, self.test_model_input, device="cpu")