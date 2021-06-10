import os
import pickle

from HParams import HParams
from Util import Util

class PreProcess():
    def __init__(self,h_params:HParams):
        self.h_params = h_params
        self.util = Util(h_params)
        self.data_log = dict()
        self.data_log_init()

    def data_log_init(self):
        self.data_log["train data num"] = 0
        self.data_log["test data num"] = 0

        self.data_log["train data time_dim_num"] = 0
        self.data_log["test data time_dim_num"] = 0
    
    def preprocess_data(self,data_name,output_path):
        meta_data_list = self.get_train_test_meta_data_list(data_name)
        
        for data_type in meta_data_list:

            for i,meta_data in enumerate(meta_data_list[data_type]):
                self.data_log[f"{data_type} data num"] += 1
                print(f"preprocess {meta_data['name']} {data_type} data {i+1}/{len(meta_data_list[data_type])}")
        
        file = open(f"{output_path}/data_info.txt",'a')
        for data_name in self.data_log:
            file.write(f"{data_name} : {self.data_log[data_name]}"+'\n')
        file.close()
        
    def get_train_test_meta_data_list(self,data_name):
        train_meta_data_list = []
        test_meta_path_list = []

        path_list_path = self.h_params.data.root_path + "/" + data_name + "_path.pkl"

        with open(path_list_path,'rb') as file_reader:
            path_list = pickle.load(file_reader)
        
        for meta_data in path_list:
            if meta_data['data_type'] == "test":
                test_meta_path_list.append(meta_data)
            else:
                train_meta_data_list.append(meta_data)
        
        return {'train': train_meta_data_list, 'test': test_meta_path_list}
    
    def save_dict_feature_by_segment(self,feature_dict,full_size,segment_size,output_path,file_name):
        for start_idx in range(0,full_size,segment_size):
            end_idx = start_idx + segment_size
            segment_feature_dict = dict()

            for feature_name in feature_dict:
                segment_feature_dict[feature_name] = feature_dict[feature_name][...,start_idx:end_idx]
            
            if segment_feature_dict[list(segment_feature_dict.keys())[0]].shape[-1] != segment_size:
                continue

            save_path = os.path.join(output_path,f'{file_name}_{start_idx}.pkl')
            print(f'Saving: {save_path}')
            with open(save_path,'wb') as file_writer:
                pickle.dump(segment_feature_dict,file_writer)