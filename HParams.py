import argparse
import yaml
import os
import torch
from dataclasses import dataclass
from datetime import datetime

time_for_output = datetime.now().strftime('%y%m%d-%H%M%S')

class HParams(object):
    def __init__(self):
        self.mode = Mode()
        self.resource = Resource()
        self.data = Data()
        self.preprocess = PreProcess()
        self.dataset = DataSet()
        self.model = Model()
        self.train= Train()
        self.log = Logging()
        self.test = Test()
        self.evaluate = Evaluate()
        self.load_config()
    
    def load_config(self):
        if self.mode.config_path is None:
            return
        yaml_file = open(self.mode.config_path, 'r')
        config_dict = yaml.safe_load(yaml_file)
        self.dict_to_h_params(config_dict)
    
    def dict_to_h_params(self,h_params_dict):
        for data_class_name in h_params_dict:
            for var_name in h_params_dict[data_class_name]:
                setattr(getattr(self,data_class_name),var_name,h_params_dict[data_class_name][var_name])
    
    def get_h_params_dict(self):
        h_params_dict = dict()
        for data_class_name in self.__dict__:
            h_params_dict[data_class_name] = dict()
            data_class = self.__dict__[data_class_name]
            for var_of_data_class in data_class.__dict__:
                h_params_dict[data_class_name][var_of_data_class] = data_class.__dict__[var_of_data_class]
        return h_params_dict
    
    @staticmethod
    def get_import_path_of_module(root_path,module_name):
        path_queue = [root_path]
        while path_queue:
            path_to_search = path_queue.pop(0)
            for dir_name in os.listdir(path_to_search):
                path = path_to_search + "/" + dir_name
                if os.path.isdir(path):
                    path_queue.append(path)
                else:
                    file_name =  os.path.splitext(dir_name)[0]
                    if file_name == module_name:
                        final_path = (path_to_search + "/" + file_name).replace("./","").replace("/",".")
                        return final_path
        return None

##########################################################################################
# Data class
##########################################################################################

@dataclass
class Mode:
    config_path:str = "./Config/baseline.yaml"
    app:str = {0:"make_meta_data", 1:"preprocess", 2:"test_model_io", 3:"train", 4:"test", 5:"evaluate"}[0]
    train:str = ["start","resume"][0]
    debug_mode:bool = False

@dataclass
class Resource:
    num_workers:int = 8
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

@dataclass
class Data:
    original_data_path:str = "../210101_data"
    root_path:str = "./Data"
    name_list = []
    preprocess_data_path = "Preprocessed"
    #data1: str = 'musdb_main_vocal'
    #name_list.append(data1)

@dataclass
class PreProcess:
    pass

@dataclass
class DataSet:
    root_path:str = "./LoadData"
    make_valid_set_from_train:bool = False
    valid_ratio:float = 0.1

@dataclass
class Model:
    root_path:str = "./Model"
    
@dataclass
class Train:
    root_path:str = "./Trainer"
    seed_strict = False
    seed = (int)(torch.cuda.initial_seed() / (2**32))
    batch_size:int = 32
    lr:int = 0.001
    lr_decay:float = 0.98
    lr_decay_step:float = 1.0E+3
    epoch:int = 2000
    save_model_after_epoch:int = 1000
    save_model_every_epoch:int = 200

@dataclass
class Logging():
    root_path = "./Log"
    log_path = os.path.join(root_path,time_for_output)
    log_name = os.path.join(log_path,"log.txt")
    tensorboard_path = os.path.join(log_path,"tb")
    log_every_local_step = 200

@dataclass
class Test():
    root_path:str = "./Tester"
    seg_time_length:int = 0
    test_type = ["one","set"][1]
    input_path = "./TestInput"
    input_file_name = ""
    input_file_ext = ".wav"
    output_path = "./TestOutput"
    pretrain_path = "./Pretrained"
    pretrain_dir_name = ""
    pretrain_module_name="pretrained_"

@dataclass
class Evaluate():
    root_path:str = "./Evaluater"
    pretrain_name = ""