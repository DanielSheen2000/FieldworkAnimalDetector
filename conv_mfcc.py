'''After reading numerous journals and tutorials online,
   including past ECE324 final project reports about MFCC,
   we decided to use the python librosa package to calculate the MFCC for us
   General steps taken:
      1) Convert YouTube videos into .wav files using youtube-dl package
      2) Normalize the audio data
      3) Convert to MFCC using Librosa package
   See more about the package: https://librosa.org/doc/latest/index.html
'''

import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import wave
import sys
import os

def normalize_audio(audio):
    audio = audio / np.max(np.abs(audio))
    return audio

#Extract mfcc features(mfcc, first time derivative, second time derivative) from the audio
def wav_to_mfcc(audio, sample_rate, **kwargs):
  audio = normalize_audio(audio)
  mfccs = librosa.feature.mfcc(audio, sample_rate, n_mfcc = 13,**kwargs)
  delta = librosa.feature.delta(mfccs)
  delta_delta = librosa.feature.delta(mfccs, order = 2)
  return np.array([mfccs,delta,delta_delta])

#Create a mfcc plot
def plot_mfcc(mfcc):
  librosa.display.specshow(mfcc, x_axis = 'time')
  plt.colorbar()
  plt.title('Normalized MFCC')
  plt.tight_layout()
  plt.show()

#Create a plot of the original sound signal
def plot_wav(audio,sample_rate):
  audio = normalize_audio(audio)
  plt.figure(figsize=(15,4))
  plt.plot(np.linspace(0, len(audio) / sample_rate, num=len(audio)), audio)
  plt.grid(True)
  plt.show()

'''The following code will extract MFCC features from each sound files in the raw data folder
   with the specified parameters: n_mfcc = 13, 20ms fft window length, 10ms fft hop length;
   Output will be saved in the processed data folder as numpy arrays of form (mfcc, delta, delta_delta)
'''
for files in os.listdir('data/raw data/hyenas/'):
  print(files)
  audio,sample_rate = librosa.load('data/raw data/hyenas/'+files)
  #Will be computing mfcc features for each frame with length of 10ms
  n_fft = int(sample_rate*0.02)
  #hop_length = # of samples bet'n each frame
  hop_length = n_fft//2
  features = wav_to_mfcc(audio,sample_rate,n_fft = n_fft,hop_length=hop_length)
  np.save('data/processed data/hyenas/'+files,features)
print('hyenas done')

for files in os.listdir('data/raw data/lions/'):
  print(files)
  audio, sample_rate = librosa.load('data/raw data/lions/' + files)
  # Will be computing mfcc features for each frame with length of 10ms
  n_fft = int(sample_rate * 0.02)
  # hop_length = # of samples bet'n each frame
  hop_length = n_fft // 2
  features = wav_to_mfcc(audio, sample_rate, n_fft=n_fft, hop_length=hop_length)
  np.save('data/processed data/lions/' + files, features)
print('Lions Done')

print('Finished extracting MFCCs')