import os
import pickle
from HParams import HParams

class MakeMedlyDbVocalRawMix:
    def __init__(self,h_params:HParams):
        self.h_params = h_params
        self.data_root = os.path.join(self.h_params.data.original_data_path,"MedlyDbVocalRawMix")

        self.accom_path = os.path.join(self.data_root,"accom")
        self.chorus_raw_path = os.path.join(self.data_root,"chorus_raw")
        self.main_vocal_raw_path = os.path.join(self.data_root,"main_vocal_raw")
        self.main_vocal_mix_path = os.path.join(self.data_root,"main_vocal_mix")

        self.test_song_list = open(self.data_root + "/" + "test_data_list.txt", 'r').read().split('\n')
    
    def save_meta_data_list(self):
        meta_data_list = []
        for main_raw_vocal in os.listdir(self.main_vocal_raw_path):
            song_name = main_raw_vocal.split("_main_vocal")[0]
            data_type = "test" if song_name in self.test_song_list else "train"
            main_vocal_raw_path = os.path.join(self.main_vocal_raw_path,main_raw_vocal)
            main_vocal_mix_path = main_vocal_raw_path.replace("raw","mix")
            chorus_raw_path = os.path.join(self.chorus_raw_path,song_name+"_chorus_raw.wav")
            if os.path.exists(chorus_raw_path) is False:
                chorus_raw_path = None
            accom_path = os.path.join(self.accom_path,song_name+"_accom.wav")
            meta_data = {   'name':song_name,
                            'main_vocal_raw':main_vocal_raw_path,
                            'main_vocal_mix':main_vocal_mix_path,
                            'chorus_raw':chorus_raw_path,
                            'accom':accom_path,
                            'data_type':data_type}
            meta_data_list.append(meta_data)
        
        save_path = self.h_params.data.root_path + "/medly_db_vocal_raw_mix_path.pkl"
        with open(save_path,'wb') as file_writer:
            pickle.dump(meta_data_list,file_writer)