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