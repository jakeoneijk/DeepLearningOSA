import os
from abc import ABC,abstractmethod

from HParams import HParams

class Evaluater(ABC):
    def __init__(self, h_params:HParams):
        self.h_params = h_params
        self.output_dir = self.h_params.test.output_path
        self.evaluation_pretrained_name = self.h_params.evaluate.pretrain_name
    
    '''
    ==============================================================
    abstract method start
    ==============================================================
    '''
    @abstractmethod
    def read_pred_gt_list(self,data_name):
        '''
        return {"pred": list, "gt" : list}
        '''
        pass

    @abstractmethod
    def evaluator(self,test_set_dict):
        '''
        return evaluation resutl
        '''
        pass
    '''
    ==============================================================
    abstract method end
    ==============================================================
    '''
    
    def process(self):
        evaluation_results_dict = dict()
        for data_name in os.listdir(f"{self.output_dir}/{self.evaluation_pretrained_name}"):
            test_set_dict:dict = self.read_pred_gt_list(self,data_name)
            evaluation_result = self.evaluator(test_set_dict)
            evaluation_results_dict = self.append_result(evaluation_results_dict,evaluation_result)
            feature_of_results = self.extract_feature_from_results(evaluation_results_dict)
    

    def append_result(self,results_dict, new_result_dict):
        pass

    def extract_feature_from_results(self,results_dict):
        '''
        mean max min for each feature
        '''
        pass

    def report_and_save_result(self,result_dict):
        '''
        print and yaml save
        '''
        pass

    

