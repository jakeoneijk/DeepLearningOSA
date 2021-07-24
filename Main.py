import os
import pickle
import importlib

from HParams import HParams
from MakeMetaData.MakeMetaData import MakeMetaData
from PreProcess import PreProcess
from TestModelIO import TestModelIO
from LoadData.DataLoaderLoader import DataLoaderLoader

class Controller():
    def __init__(self):
        self.h_params:HParams = None

        self.model = None
        self.trainer = None
        self.tester = None
        self.evaluater = None

    def set_experiment(self,h_params:HParams = None):
        #set hparams.
        self.h_params = HParams() if h_params is None else h_params
        os.makedirs(self.h_params.test.output_path,exist_ok=True)
        self.set_experiment_module()
    
    def set_experiment_module(self):
        #set model, trainer and tester
        if self.h_params.mode.app in ["test_model_io", "train", "test"]:
            module_path = self.h_params.get_import_path_of_module(self.h_params.model.root_path, self.h_params.model.name)
            model_module= importlib.import_module(module_path)
            self.model = getattr(model_module,self.h_params.model.name)(self.h_params)
            
        if self.h_params.mode.app == "train":
            module_path = self.h_params.get_import_path_of_module(self.h_params.train.root_path, self.h_params.train.name)
            trainer_module= importlib.import_module(module_path)
            self.trainer = getattr(trainer_module,self.h_params.train.name)(self.model,self.h_params)

        if self.h_params.mode.app == "test":
            module_path = self.h_params.get_import_path_of_module(self.h_params.test.root_path, self.h_params.test.name)
            tester_module= importlib.import_module(module_path)
            self.tester = getattr(tester_module,self.h_params.test.name)(self.model,self.h_params)

        if self.h_params.mode.app == "evaluate":
            module_path = self.h_params.get_import_path_of_module(self.h_params.evaluate.root_path, self.h_params.evaluate.name)
            evaluater_module= importlib.import_module(module_path)
            self.evaluater = getattr(evaluater_module,self.h_params.evaluate.name)(self.h_params)

    def run(self):
        print("=============================================")
        print(f"{self.h_params.mode.app} start")
        print("=============================================")
        
        if self.h_params.mode.app == "make_meta_data":
            self.make_meta_data()

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

    def make_meta_data(self):
        meta_data_maker = MakeMetaData(self.h_params)
        for data_name in self.h_params.data.name_list:
            meta_data_maker.make_meta_data(data_name)

    def preprocess(self):
        preprocessor = PreProcess(self.h_params)
        for data_name in self.h_params.data.name_list:
            preprocessor.preprocess_data(data_name)
    
    def test_model_io(self):
        test_moder_io = TestModelIO(self.model.cpu(),self.h_params)
        test_moder_io.test()

    def train(self):
        data_loader_loader = DataLoaderLoader(self.h_params)
        train_data_loader,valid_data_loader,test_data_loader = data_loader_loader.get_data_loader()
        """
        construct trainer here and fit
        """
        self.trainer.set_data_loader(train_data_loader,valid_data_loader,test_data_loader)
        self.trainer.fit()

    def test(self):
        with open(self.h_params.test.pretrain_path+"/"+self.h_params.test.pretrain_dir_name+"/h_params.pkl",'rb') as pickle_file:
            h_params_of_pretrained:HParams = pickle.load(pickle_file)
        self.h_params.model = h_params_of_pretrained.model
        self.h_params.preprocess = h_params_of_pretrained.preprocess
        for data_name in self.h_params.data.name_list:
            self.tester.test_test_set(data_name)

    def evaluate(self):
        self.evaluater.process()

if __name__ == '__main__':
    controller = Controller()
    controller.set_experiment()
    controller.run()