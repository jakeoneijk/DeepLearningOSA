import os
import yaml
import soundfile as sf

class MedlyDBForSingingVoiceSepatation():
    def __init__(self):
        self.stem_file_path = '/media/bach3/dataset/MedleyDB/Audio'
        self.file_name_list = os.listdir(self.stem_file_path)
        self.output_path = "./MedlyDBSVS"
        self.output_mix_path = os.path.join(self.output_path,"mix")
        self.output_vocal_accom_path = os.path.join(self.output_path,"vocal_accom")
        os.makedirs(self.output_path,exist_ok=True)
        os.makedirs(self.output_mix_path,exist_ok=True)
        os.makedirs(self.output_vocal_accom_path,exist_ok=True)
        
    
    def make_data(self):
        instrument_list = []
        for i,audio in enumerate(self.file_name_list):
            print(f'{i}/{len(self.file_name_list)}')
            vocal_stem_num = []
            inst_stem_num = []
            if audio == "AmarLal_SpringDay1":
                print("de")
            if 'DS_Store' in audio:
                continue

            folder_path = os.path.join(self.stem_file_path,audio)
            meta_file_path = folder_path + "/" + audio + "_METADATA.yaml"
            with open(meta_file_path) as f:
                meta_data = yaml.load(f)
            
            for stem_key in meta_data['stems']:

                if self.is_vocal_stem_file(meta_data['stems'][stem_key]):
                    vocal_stem_num.append(stem_key.replace("S",""))
                else:
                    inst_stem_num.append(stem_key.replace("S",""))

            if len(vocal_stem_num) == 0 or len(inst_stem_num) == 0:
                print("no vocal or stem")
                continue
            
            vocal_stem_path = [folder_path + "/" + audio + "_STEMS/"+audio+"_STEM_"+num+".wav" for num in vocal_stem_num]
            accom_stem_path = [folder_path + "/" + audio + "_STEMS/"+audio+"_STEM_"+num+".wav" for num in inst_stem_num]
            
            vocal = None
            accom = None

            for vocal_path in vocal_stem_path:
                x, sr = sf.read(vocal_path)
                vocal = x if vocal is None else vocal + x

            for accom_path in accom_stem_path:
                x, sr = sf.read(accom_path)
                accom = x if accom is None else accom + x
            
            mix = vocal + accom
            sf.write(self.output_vocal_accom_path+"/"+audio+"_vocal.wav",vocal,sr)
            sf.write(self.output_vocal_accom_path+"/"+audio+"_accom.wav",accom,sr)
            sf.write(self.output_mix_path+"/"+audio+"_mix.wav",mix,sr)
                

                
                
        
    def is_vocal_stem_file(self,stem_data):
        if 'male' in stem_data['instrument']:
            return True
        if 'vocal' in stem_data['instrument']:
            return True
        return False


if __name__ == '__main__':
    medly = MedlyDBForSingingVoiceSepatation()
    medly.make_data()
