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
        self.model = Model()
        self.train= Train()
        self.log = Logging()
        self.test = Test()
        self.config = Config()
        self.load_config()

        self.make_essential_dir()
    
    def load_config(self):
        if self.config.path is None:
            return
        yaml_file = open(self.config.path, 'r')
        yaml_dict = yaml.safe_load(yaml_file)
        
        for key in yaml_dict:
            data_class_and_vars_name = key.split("~")
            if len(data_class_and_vars_name) <= 1:
                continue
            data_class_name = data_class_and_vars_name[0]
            vars_name = data_class_and_vars_name[1]
            setattr(getattr(self,data_class_name),vars_name,yaml_dict[key])

        self.config.config = yaml_dict

    def make_essential_dir(self):
        os.makedirs(self.data.root_path,exist_ok=True)
        for data_name in self.data.name_list:
            data_path = os.path.join(self.data.root_path,data_name)
            os.makedirs(data_path,exist_ok=True)
            os.makedirs(data_path+"/"+self.data.preprocess_data_path,exist_ok=True)
            os.makedirs(data_path+"/"+self.data.preprocess_data_path+"/train",exist_ok=True)
            os.makedirs(data_path+"/"+self.data.preprocess_data_path+"/test",exist_ok=True)

        if self.mode.app == "train":
            os.makedirs(self.log.log_root_path,exist_ok=True)
            os.makedirs(self.log.log_path,exist_ok=True)
            os.makedirs(self.log.tensorboard_path,exist_ok=True)
            os.makedirs(self.log.model_save_path,exist_ok=True)
        
        os.makedirs(self.test.pretrain_path,exist_ok=True)
        os.makedirs(self.test.output_path,exist_ok=True)

##########################################################################################
# Data class
##########################################################################################

@dataclass
class Mode:
    experiment_name:str = "base_line"
    app = ["make_meta_data", "preprocess", "test_model_io", "train", "test", "evaluate"][0]
    train:str = ["start","resume"][0]
    debug_mode:bool = True

@dataclass
class Resource:
    num_workers = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class Data:
    valid_ratio = 0.1
    original_data_path = "../210101_data"
    root_path = "./Data"
    name_list = []
    preprocess_data_path = "Preprocessed"
    #data1: str = 'sk_multi_singer'
    #name_list.append(data1)

@dataclass
class Config:
    path:str = "./Config/baseline.yaml"
    config:dict = None

@dataclass
class PreProcess:
    sample_rate:int = 44100

@dataclass
class Model:
    pass

@dataclass
class Train:
    seed_strict = False
    seed = (int)(torch.cuda.initial_seed() / (2**32))
    batch_size:int = 16
    lr:int = 0.001
    epoch:int = 1000

@dataclass
class Logging():
    log_root_path = "./Log"
    log_path = os.path.join(log_root_path,time_for_output)
    log_name = os.path.join(log_path,"log.txt")
    tensorboard_path = os.path.join(log_path,"tb")
    model_save_path = log_path
    model_save_name = ""
    log_every_local_step = 200

@dataclass
class Test():
    seg_time_length:int
    test_type = ["one","set"][1]
    test_set_list_path = './test_song_list.txt'
    data_type = ''
    input_path = "./TestInput"
    input_file_name = ""
    input_file_ext = ".wav"
    output_path = "./TestOutput"
    pretrain_path = "./Pretrained"
    pretrain_dir_name = ""