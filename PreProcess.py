import os
import pickle

from HParams import HParams
from Util import Util

class PreProcess():
    def __init__(self,h_params:HParams):
        self.h_params = h_params
        self.util = Util(h_params)
    
    def preprocess_data(self,data_name,output_path):
        path_list_path = self.h_params.data.root_path + "/" + data_name + "_path.pkl"
        with open(path_list_path,'rb') as file_reader:
            path_list = pickle.load(file_reader)
        
        test_song_list = open(self.h_params.data.root_path + "/" + data_name+"_test_data_list.txt", 'r').read().split('\n')
        for i,song_data in enumerate(path_list):
            print(f"preprocess {song_data['name']} {i+1}/{len(path_list)}")
            data_type =  'test' if song_data['name'] in test_song_list else 'train'
    
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