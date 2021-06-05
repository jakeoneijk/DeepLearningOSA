import os
import pickle

from HParams import HParams
from PreProcess import PreProcess
from TestModelIO import TestModelIO
from LoadData.DataLoaderLoader import DataLoaderLoader

class Controller():
    def __init__(self):
        self.h_params:HParams = None

    def set_hparams(self):
        #set hparams.
        self.h_params = HParams()

    def run(self):
        print("=============================================")
        print(f"{self.h_params.mode.app} start")
        print("=============================================")
        
        if self.h_params.mode.app == "make_data":
            self.make_data()

        if self.h_params.mode.app == "preprocess":
            self.preprocess()

        if self.h_params.mode.app == "test_model_io":
            self.test_model_io()
        
        if self.h_params.mode.app == "train":
            self.train()
        
        if self.h_params.mode.app == "test":
            self.test()

        if self.h_params.mode.app == "evaluate":
            self.evaluate()
        
        print("finish app")

    def make_data(self):
        pass

    def preprocess(self):
        for data_name in self.h_params.data.name_list:
            data_output_root = os.path.join(self.h_params.data.root_path,data_name) + "/Preprocessed"
            data_root = self.h_params.data.original_data_path + "/"+data_name
    
    def test_model_io(self):
        test_moder_io = TestModelIO(self.h_params)
        test_moder_io.test()

    def train(self):
        data_loader_loader = DataLoaderLoader(self.h_params)
        train_data_loader,valid_data_loader,test_data_loader = data_loader_loader.get_data_loader()
        """
        construct trainer here and fit
        """
        trainer = None
        trainer.set_data_loader(train_data_loader,valid_data_loader,test_data_loader)
        trainer.fit()

    def test(self):
        with open(self.h_params.test.pretrain_path+"/"+self.h_params.test.pretrain_dir_name+"/h_params.pkl",'rb') as pickle_file:
            h_params_of_pretrained = pickle.load(pickle_file)
        self.h_params.model = h_params_of_pretrained.model
        self.h_params.preprocess = h_params_of_pretrained.preprocess
        pass

    def evaluate(self):
        pass

if __name__ == '__main__':
    controller = Controller()
    controller.set_hparams()
    controller.run()