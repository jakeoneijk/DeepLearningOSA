import torch
from torchsummary import summary
from HParams import HParams

class TestModelIO():
    def __init__(self,h_params:HParams):
        self.test_model = None
        self.test_model_input = torch.randn((1)).cuda()
    
    def test(self):
        summary(self.test_model, (self.channel,self.time_dimenstion))
        output = self.test_model(self.test_model_input)
        print(output.size())