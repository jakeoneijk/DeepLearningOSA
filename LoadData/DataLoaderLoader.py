import os
import random
import math

from torch.utils.data import DataLoader

from HParams import HParams
from LoadData.DataSet import DataSet

class DataLoaderLoader():
    def __init__(self,h_params:HParams):
        self.h_params = h_params
    
    def get_data_set(self,path_list):
        return DataSet(path_list)
    
    def get_data_loader(self):
        train_path_list,valid_path_list,test_path_list = self.get_data_path_list()

        train_data_set = self.get_data_set(train_path_list)
        valid_data_set = self.get_data_set(valid_path_list)
        test_data_set = self.get_data_set(test_path_list)

        train_data_loader = DataLoader(train_data_set,pin_memory=True,batch_size=self.h_params.train.batch_size, 
        shuffle=True, num_workers=self.h_params.resource.num_workers, drop_last=True)
        valid_data_loader = DataLoader(valid_data_set,pin_memory=True,batch_size=self.h_params.train.batch_size, 
        shuffle=False, num_workers=self.h_params.resource.num_workers, drop_last=True)
        test_data_loader = DataLoader(test_data_set, batch_size=self.h_params.train.batch_size, 
        shuffle=False, num_workers=self.h_params.resource.num_workers, drop_last=False)

        return train_data_loader,valid_data_loader,test_data_loader

    def get_data_path_list(self):
        total_train_path_list = []
        total_valid_path_list = []
        total_test_path_list = []
        for data_name in self.h_params.data.name_list:
            data_path = os.path.join(self.h_params.data.root_path,data_name+"/"+self.h_params.data.preprocess_data_path) 
            train_path_list,valid_path_list,test_path_list = self.get_data_path(data_path)
            total_train_path_list = total_train_path_list + train_path_list
            total_valid_path_list = total_valid_path_list + valid_path_list
            total_test_path_list = total_test_path_list + test_path_list
        
        if self.h_params.mode.debug_mode:
            print("use small data because of debug mode")
            total_train_path_list = total_train_path_list[:20]
            total_valid_path_list = total_valid_path_list[:20]
            total_test_path_list = total_test_path_list[:20]
        
        return total_train_path_list, total_valid_path_list, total_test_path_list
    
    def get_data_path(self,data_path,make_valid_set_from_train = False):
        total_train_data_list = [os.path.join(data_path+"/train",fname) for fname in os.listdir(data_path+"/train")]
        test_data_list = [os.path.join(data_path+"/test",fname) for fname in os.listdir(data_path+"/test")]
        
        if make_valid_set_from_train == False:
            valid_data_list = [os.path.join(data_path+"/valid",fname) for fname in os.listdir(data_path+"/valid")]
            return total_train_data_list,valid_data_list,test_data_list
        
        #divide train and valid
        num_total_train_data = len(total_train_data_list)
        total_indices = list(range(num_total_train_data))
        random.shuffle(total_indices)
        num_train_set = math.floor(num_total_train_data * (1-self.h_params.data.valid_ratio))
        train_idx = total_indices[:num_train_set]
        valid_idx = total_indices[num_train_set:]

        train_data_list = [total_train_data_list[i] for i in train_idx]
        valid_data_list = [total_train_data_list[i] for i in valid_idx]

        return train_data_list,valid_data_list,test_data_list
