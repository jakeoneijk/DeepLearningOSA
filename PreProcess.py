import os
import pickle
import yaml
import numpy as np
from HParams import HParams
from Util.Util import Util

class PreProcess():
    def __init__(self,h_params:HParams):
        self.h_params = h_params
        self.util = Util(h_params)
        self.data_log = dict()
        self.preprocessed_folder_name = "Preprocessed"

    def data_log_init(self,data_path):
        os.makedirs(data_path,exist_ok=True)
        os.makedirs(f"{data_path}/{self.preprocessed_folder_name}",exist_ok=True)
        for data_type in ["train","valid","test"]:
            self.data_log[f"{data_type} data num"] = 0
            self.data_log[f"{data_type} time_dim_num"] = 0
            os.makedirs(f"{data_path}/{self.preprocessed_folder_name}/{data_type}",exist_ok=True)
        self.data_log["train feature min max"] = dict()
    
    def feature_min_max_log(self,feature_dict):
        for feature_name in feature_dict:
            if type(feature_dict[feature_name]) not in [list,np.ndarray]:
                    continue
            if feature_name not in self.data_log["train feature min max"]:
                self.data_log["train feature min max"][feature_name] = {"min": np.inf, "max": -np.inf}
            self.data_log["train feature min max"][feature_name]["min"] = min(self.data_log["train feature min max"][feature_name]["min"],
                                                                        np.min(feature_dict[feature_name]))
            self.data_log["train feature min max"][feature_name]["max"] = max(self.data_log["train feature min max"][feature_name]["max"],
                                                                        np.max(feature_dict[feature_name]))
        
    def preprocess_data(self,data_name):
        data_path = os.path.join(self.h_params.data.root_path,data_name)
        self.data_log_init(data_path)
        
        meta_data_list = self.get_train_test_meta_data_list(data_name)
        
        for i,meta_data in enumerate(meta_data_list):
            self.data_log[f"{meta_data['data_type']} data num"] += 1
            print(f"preprocess {meta_data['name']} {i+1}/{len(meta_data_list)}")
            feature_dict = self.preprocess_one_data(data_name,meta_data)

            feature_dict["segment_dim_size"] = self.dict_dim_size_consistency_check(feature_dict)
            
            self.data_log[f"{meta_data['data_type']} time_dim_num"] += feature_dict["segment_dim_size"]

            self.save_dict_feature_by_data_name_folder( feature_dict = feature_dict,
                                                        output_path = f"{data_path}/{self.preprocessed_folder_name}",
                                                        meta_data = meta_data)
            if meta_data["data_type"] == "train":
                self.feature_min_max_log(feature_dict)
                
        with open(f"{data_path}/{self.preprocessed_folder_name}/data_info.txt",'w') as file:
            yaml.dump(self.data_log, file)
    
        
    def get_train_test_meta_data_list(self,data_name):
        path_list_path = self.h_params.data.root_path + "/" + data_name + "_path.pkl"
        with open(path_list_path,'rb') as file_reader:
            path_list = pickle.load(file_reader)
        
        return path_list

    def save_dict_feature_by_data_name_folder(self,feature_dict,output_path,meta_data):
        preprocessed_path = f"{output_path}/{meta_data['data_type']}/{meta_data['name']}"
        os.makedirs(preprocessed_path)
        for feature_name in feature_dict:

            if type(feature_dict[feature_name]) not in [list,np.ndarray]:
                continue

            save_path = os.path.join(preprocessed_path,f'{feature_name}.pkl')
            print(f'Saving: {save_path}')
            with open(save_path,'wb') as file_writer:
                pickle.dump(feature_dict[feature_name],file_writer)
    
    def dict_dim_size_consistency_check(self,input_dict:dict,dim_axis=-1):
        dim_size = None
        for key in input_dict:
            if dim_size is None:
                dim_size = input_dict[key].shape[dim_axis]
            else:
                assert(dim_size == input_dict[key].shape[dim_axis]),"check time dim size of mel and stft"
        return dim_size
    
    def preprocess_one_data(self,data_name,meta_data):
        pass