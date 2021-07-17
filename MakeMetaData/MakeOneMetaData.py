import os
import pickle
from HParams import HParams

from abc import ABC, abstractmethod

class MakeOneMetaData(ABC):
    def __init__(self,h_params:HParams):
        self.h_params = h_params
        self.data_name = self.get_data_name()
        self.data_root = self.get_data_root()
    '''
    ==============================================================
    abstract method start
    ==============================================================
    '''
    @abstractmethod
    def get_data_name(self):
        '''
        return data_name    : str
        '''
        raise NotImplementedError
    
    @abstractmethod
    def get_data_root(self):
        '''
        return original data_root    : str
        '''
        raise NotImplementedError
    
    @abstractmethod
    def get_meta_data(self):
        '''
        return meta_data_list   : list
        each metadata is dict and must have
        'name'
        'data_type' (train test valid)
        'sample_rate' if audio
        path of the data
        '''
        raise NotImplementedError
    '''
    ==============================================================
    abstract method end
    ==============================================================
    '''
    def save_meta_data_list(self):
        meta_data_list = self.get_meta_data()
        save_path =  f"{self.h_params.data.root_path}/{self.data_name}_path.pkl"
        with open(save_path,'wb') as file_writer:
            pickle.dump(meta_data_list,file_writer)
