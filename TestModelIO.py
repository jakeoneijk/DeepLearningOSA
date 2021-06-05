from HParams import HParams
class TestModelIO():
    def __init__(self,h_params:HParams):
        self.test_model = None
        self.test_model_input = torch.randn((1)).to(h_params.resource.device)
    
    def test(self):
        output = self.test_model(self.test_model_input)
        print(output.size())