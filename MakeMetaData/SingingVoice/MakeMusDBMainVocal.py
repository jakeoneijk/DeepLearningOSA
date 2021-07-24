import os
import yaml
from HParams import HParams
from MakeMetaData.MakeOneMetaData import MakeOneMetaData

class MakeMusDBMainVocal(MakeOneMetaData):
    def __init__(self,h_params:HParams):
        super().__init__(h_params)
        self.sr = 44100
        self.validation_tracks_list = yaml.safe_load(open(self.data_root+"/musdb.yaml",'r'))['validation_tracks']
    
    def get_data_name(self):
        '''
        return data_name    : str
        '''
        return "musdb_main_vocal"
    
    def get_data_root(self):
        '''
        return original data_root    : str
        '''
        return os.path.join(self.h_params.data.original_data_path,"MusDBMainVocal")
    
    def get_meta_data(self):
        '''
        return meta_data_list   : list
        each metadata is dict and must have
        'name'
        'data_type' (train test valid)
        path of the data
        '''
        meta_data_list = []
        for data_type in ['train','test']:
            for song_name in os.listdir(f"{self.data_root}/{data_type}"):
                meta_data = dict()
                meta_data['name'] = song_name
                meta_data['data_type'] = 'valid' if song_name in self.validation_tracks_list else data_type
                meta_data['sample_rate'] = 44100
                for stem in os.listdir(f"{self.data_root}/{data_type}/{song_name}"):
                    stem_type = os.path.splitext(stem.split("_")[-1])[0]
                    meta_data[stem_type] = f"{self.data_root}/{data_type}/{song_name}/{stem}"
                
                meta_data_list.append(meta_data)
        
        return meta_data_list