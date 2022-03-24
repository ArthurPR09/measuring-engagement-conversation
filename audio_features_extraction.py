import os
import numpy as np
import ffmpeg
import librosa

cheese_folder = "//filer/AGORA/Comprendre/CHEESE/Audio"
paco_folder = "//filer/AGORA/Comprendre/PACO/Audio "

all_audio_features = {}

all_pitch = []
for folder in [cheese_folder, paco_folder]:
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            with open(file, 'r') as f:
                praat_pitch = f.readlines()
            pitch = []
            for i in range(1, len(praat_pitch)):
                if "candidate [1]" in praat_pitch[i-1]:
                    pitch.append([int(praat_pitch[i-5].split('[')[1][:-3]),
                                  float(praat_pitch[i].split('=')[1])])
            all_pitch.append(np.array(pitch))

window = 0.035 #?
stride = 0.01
cnt = 0
for subdir, dirs, files in os.walk(folder):
    for file in files:
        audio_features = {'Energy': None, 'Log-energy': None, 'RMSE': None, 'MFCCs': None}
        sig, sr = librosa.load(file, sr=None)
        audio_features['Energy'] = np.array(sum([sig[i:i+window*sr]**2 for i in range(0, len(sig), int(stride * sr))]))
        audio_features['Log-energy'] = np.log(audio_features['Energy'])
        audio_features['RMSE'] = np.sqrt(audio_features['Energy'] / (window * sr))
        audio_features['MFCCs'] = librosa.feature.mfcc(sig, sr=sr, n_mfcc=13, hop_length=int(stride * sr),
                                                       n_fft=int(window * sr))
        audio_features['Pitch'] = all_pitch[cnt]
        all_audio_features[...] = audio_features
        cnt += 1

