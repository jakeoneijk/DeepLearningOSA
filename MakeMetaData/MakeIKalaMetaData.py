import os
import pickle
from HParams import HParams

class MakeIKalaMetaData:
    def __init__(self,h_params:HParams):
        self.h_params = h_params
        self.data_root = os.path.join(self.h_params.data.original_data_path,"iKala")
        self.wavfile_root = os.path.join(self.data_root,"Wavfile")
    
    def save_meta_data_list(self):
        meta_data_list = []
        test_song_list = open(self.data_root + "/" + "test_data_list.txt", 'r').read().split('\n')
        for mix_song in os.listdir(self.wavfile_root):
            song_name = mix_song.replace(".wav","")
            data_type = "test" if song_name in test_song_list else "train"
            mix_song_path = os.path.join(self.wavfile_root,mix_song)
            song_data = {"name":song_name,"mix":mix_song_path,'data_type':data_type}
            meta_data_list.append(song_data)
        save_path = self.h_params.data.root_path + "/ikala_path.pkl"
        with open(save_path,'wb') as file_writer:
            pickle.dump(meta_data_list,file_writer)