import torch.utils.data.dataset as dataset
import pickle
from random import randint

from HParams import HParams

class FeatureSegmentDataSet(dataset.Dataset):
    '''
    segment dim is -1 (last)
    '''
    def __init__(self, h_params:HParams, data_path_list:list):
        self.batch_size = h_params.train.batch_size
        self.segment_size = h_params.model.segment_size

        self.data_set = []
        for data_path in data_path_list:
            feature_dict = dict()
            for feature_name in h_params.dataset.feature_list:
                feature_dict[feature_name] = self.read_feature_pickle(f"{data_path}/{feature_name}.pkl")
            self.data_set.append(feature_dict)
    
    def read_feature_pickle(self, data_path):
        with open(data_path, 'rb') as pickle_file:
            feature = pickle.load(pickle_file)
        return feature
    
    def __len__(self):
        return self.batch_size * len(self.data_set)
    
    def __getitem__(self, index):
        index = index//self.batch_size

        segment_idx = randint(0, self.data_set[index]['segment_dim_size'] - self.segment_size)

        segmented_data = dict()

        for feature_name in self.data_set[index]:
            if feature_name == 'segment_dim_size':
                continue
            segmented_data[feature_name] = self.data_set[index][feature_name][...,segment_idx:segment_idx+self.segment_size]
        
        return segmented_data