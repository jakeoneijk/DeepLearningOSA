import os
import pickle
from HParams import HParams

class MakeSkMultiSingerMetaData:
    def __init__(self,h_params:HParams):
        self.h_params = h_params
        self.data_root = os.path.join(self.h_params.data.original_data_path,"SkMultiSinger")
    
    def save_meta_data_list(self):
        meta_data_list = []

        mix_data_root = os.path.join(self.data_root,"mix_by_adding")
        test_song_list = open(self.data_root + "/" + "test_data_list.txt", 'r').read().split('\n')
        
        for mix_song in os.listdir(mix_data_root):
            song_name = mix_song.replace("_mix.wav","")
            mix_song_path = os.path.join(mix_data_root,mix_song)
            vocal_song_path = os.path.join(self.data_root,song_name+"(vox).wav")
            accom_song_path = os.path.join(self.data_root,song_name+"(inst).wav")
            data_type = "test" if song_name in test_song_list else "train"
            meta_data = {'name':song_name,'mix':mix_song_path,'vocal':vocal_song_path,'accom':accom_song_path,'data_type':data_type}
            meta_data_list.append(meta_data)
        
        save_path = self.h_params.data.root_path + "/sk_multi_singer_path.pkl"
        with open(save_path,'wb') as file_writer:
            pickle.dump(meta_data_list,file_writer)