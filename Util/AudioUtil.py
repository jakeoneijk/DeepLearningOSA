import librosa
import soundfile as sf
import numpy as np

from HParams import HParams
class AudioUtil:
    def __init__(self,h_params:HParams):
        self.h_params = h_params
    
    def librosa_load_audio_fixed_sr(self,audio_path):
        audio,_ = librosa.load(audio_path,sr=self.h_params.preprocess.sample_rate)
        return audio
    
    def save_audio_fixed_sample_rate(self,path,audio):
        sf.write(f"./TestOutput/{path}.wav",audio,samplerate=self.sample_rate)
    
    def mag_phase_stft(self,audio):
        stft = librosa.stft(audio,n_fft=self.h_params.preprocess.nfft, hop_length=self.h_params.preprocess.hopsize)
        mag = abs(stft)
        phase = np.exp(1.j * np.angle(stft))
        return {"mag":mag,"phase":phase}
    
    def get_pred_accom_by_subtract_pred_vocal(self,pred_vocal,is_pred_vocal_audio,mix_audio):
        pred_vocal_mag = pred_vocal
        if is_pred_vocal_audio:
            pred_vocal_mag = self.mag_phase_stft(pred_vocal)["mag"]
        mix_stft = self.mag_phase_stft(mix_audio)
        mix_mag = mix_stft["mag"]
        mix_phase = mix_stft["phase"]
        pred_accom_mag = mix_mag - pred_vocal_mag
        pred_accom_mag[pred_accom_mag < 0] = 0
        pred_accom = librosa.istft(pred_accom_mag*mix_phase,hop_length=self.h_params.preprocess.hopsize,length=len(mix_audio))
        return pred_accom