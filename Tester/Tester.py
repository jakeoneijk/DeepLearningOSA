import os
import numpy as np
import torch
from abc import ABC, abstractmethod
from datetime import datetime

from HParams import HParams
from PreProcess import PreProcess

class Tester(ABC):
    def __init__(self,model,h_params:HParams):
        self.h_params:HParams = h_params
        self.device = h_params.resource.device
        self.model = model
        self.pretrained_load(os.path.join(self.h_params.test.pretrain_path,self.h_params.test.pretrain_dir_name)+"/model_load.pth")
        
        self.preprocessor = PreProcess(h_params)
        self.time_log = datetime.now().strftime('%y%m%d-%H%M%S')
    
    '''
    ==============================================================
    abstract method start
    ==============================================================
    '''

    @abstractmethod
    def read_input(self,meta_data,data_name):
        '''
        '''
        pass

    @abstractmethod
    def post_processing(self,model_output,original_seg_dim_size):
        pass
    
    '''
    ==============================================================
    abstract method end
    ==============================================================
    '''

    def test_test_set(self,data_name):
        meta_data_list = self.preprocessor.get_train_test_meta_data_list(data_name)
        test_meta_data_list = [meta_data for meta_data in meta_data_list if meta_data["data_type"] == "test" ]
        for test_meta_data in test_meta_data_list:
            self.test_one_sample(test_meta_data,data_name)

    def test_one_sample(self,meta_data,data_name):
        self.set_output_path(meta_data=meta_data)
        input,segment_dim_size = self.read_input(meta_data,data_name)
        batch_input = self.make_batch(input,segment_dim_size,self.h_params.model.segment_size)
        output = self.make_output(batch_input)
        self.post_processing(output,segment_dim_size)

    def pretrained_load(self,pretrain_path):
        best_model_load = torch.load(pretrain_path,map_location='cpu')
        self.model.load_state_dict(best_model_load)
        self.model.to(self.device)
    
    def set_output_path(self,meta_data):
        dataname = meta_data['name']
        model_info = self.h_params.test.pretrain_dir_name

        self.output_path = self.h_params.test.output_path
        output_path_list = [model_info, dataname]

        for output_path_dir in output_path_list:
            self.output_path = self.output_path + "/" + output_path_dir
            os.makedirs(self.output_path,exist_ok=True)

    def make_batch(self,input:dict,segment_dim_size:int, segment_size:int):
        batch_data = dict()
        for feature in input:
            batch_data[feature]=None

        total_size = segment_dim_size

        for start_idx in range(0,total_size,segment_size):
            end_idx = start_idx + segment_size

            for feature in input:
                if type(input[feature]) not in [list,np.ndarray]:
                    continue
                feature_seg = input[feature][...,start_idx:end_idx]
            
                if feature_seg.shape[-1] != segment_size:
                    padding = segment_size - feature_seg.shape[-1]
                    if len(feature_seg.shape) == 1:
                        feature_seg =  np.pad(feature_seg,(0,padding),'constant', constant_values=0)
                    elif len(feature_seg.shape) == 2:
                        feature_seg =  np.pad(feature_seg,((0,0),(0,padding)),'constant', constant_values=0)
                    else:
                        continue
                    
                feature_seg = np.expand_dims(feature_seg,axis=0)
                batch_data[feature] = feature_seg if batch_data[feature] is None else np.vstack((batch_data[feature],feature_seg))

        return batch_data
    
    def make_output(self,batch_input):
        total_pred_features = dict()
        batch_size = 16
        for start_idx in range(0,batch_input['mix_stft'].shape[0],batch_size):
            end_idx = start_idx + batch_size
            batch_input_torch = torch.from_numpy(batch_input['mix_stft'][start_idx:end_idx,...]).to(self.device).float()
            with torch.no_grad():
                pred_features = self.model(batch_input_torch)
                for feature_name in pred_features:
                    if feature_name not in total_pred_features:
                        total_pred_features[feature_name] = None
                    total_pred_features[feature_name] = self.unzip_batch(pred_features[feature_name],total_pred_features[feature_name])
        return total_pred_features
    
    def unzip_batch(self,batch_data,unzip_data):
        numpy_batch = batch_data.detach().cpu().numpy()
        for i in range(0,numpy_batch.shape[0]):
            unzip_data = numpy_batch[i] if unzip_data is None else np.concatenate((unzip_data,numpy_batch[i]),axis=-1)
        return unzip_data