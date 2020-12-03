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