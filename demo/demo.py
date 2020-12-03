'''
A Demo that shows how the model works
'''
import os
import datetime
import torch
import shutil
import conv_mfcc as cm
import model as m
from model import Net
import librosa
import numpy as np
from pydub import AudioSegment
# Download and process a target video
def process_video(url):
    print("Downloading Target video...")
    # Download the video and its audio
    os.system('"youtube-dl -f mp4 ' + url + ' --output "demo.%(ext)s" "')
    os.system('"youtube-dl -x --audio-format wav --output "demo.%(ext)s" ' + url + '"')
    print("Downloading completed, start trimming...")

    # Trim the audio
    demo_name = "demo.wav"
    video_length = 145.0
    segment_length = 4.0
    increment = 1.0
    for i in range(1, int(video_length - segment_length - 1), int(increment)):
        t_demo = i
        command_cut = "ffmpeg -i " + demo_name + " -ss " + str(t_demo) + " -to " + str(
            t_demo + segment_length) + " -c copy demo_" + str(i) + ".wav"
        os.system(command_cut)
    print("Trimming Completed, start padding...")

    # Pad the audio file
    pad_ms = 3000
    silence = AudioSegment.silent(duration=pad_ms)
    for i in range(1, int(video_length - segment_length - 1), int(increment)):
        segment_name = "demo_" + str(i) + ".wav"
        segment = AudioSegment.from_wav(segment_name)
        padded = silence + segment + silence
        padded_name = "demo_padded_" + str(i) + ".wav"
        padded.export(padded_name, format='wav')

    print("Processing Done!")
    return int(video_length - segment_length - 1)
# move the padded data to the target directory
def move_data(length):
    print("Moving data to the processing folder...")
    path_demo = 'demo_data/demo/'
    path_padded = 'demo_data/demo_padded/'
    for i in range(1, length, 1):
        demo_name = 'demo_' + str(i) + '.wav'
        padded_name = 'demo_padded_' + str(i) + '.wav'
        shutil.move(demo_name, path_demo)
        shutil.move(padded_name, path_padded)
    print("Data successfully moved to the correct directory! ")
    return True

# Convert the padded data into mfccs
def convert_mfcc():
    print("Converting data into mfcc format...")
    min_length = 2147483647
    path_padded = 'demo_data/demo_padded/'

    for files in os.listdir(path_padded):
        audio, sample_rate = librosa.load('demo_data/demo_padded/%s' % (files))
        # Will be computing mfcc features for each frame with length of 10ms
        n_fft = int(sample_rate * 0.02)
        # hop_length = # of samples bet'n each frame
        hop_length = int(n_fft // 2)
        features = cm.wav_to_mfcc(audio, sample_rate)
        if (features.shape[2] < min_length):
            min_length = features.shape[2]
        np.save('demo_data/demo_mfcc/%s' % (files), features)
    print('Finished extracting MFCCs')
    print('Minimum length is %i' % (min_length))
    return min_length

def load_mfcc():
    print("Loading mfcc data and creating dataset...")
    mfccs = []
    deltas = []
    delta_deltas = []
    X = []
    min_length = 430
    min_shape = 1000
    path_mfcc = 'demo_data/demo_mfcc/'
    for file in os.listdir(path_mfcc):
        demo_mfcc = np.load(path_mfcc + file)
        if min_shape > demo_mfcc[0].shape[1]:
            min_shape = demo_mfcc[0].shape[1]
        mfcc = np.asarray(demo_mfcc[0][:, :min_length])
        delta = np.asarray(demo_mfcc[1][:, :min_length])
        delta_delta = np.asarray(demo_mfcc[2][:, :min_length])
        mfccs.append(mfcc)
        deltas.append(delta)
        delta_deltas.append(delta_delta)
        X.append([mfcc, delta, delta_delta])

    print("Dataset generated successfully! ")
    return X

# Load the trained model and get prediction data
def apply_model(datas):
    print("Applying data to the trained model, loading the model...")
    # Load the trained model
    model = torch.load('model.pt')
    model.eval()
    print("Model loaded successfully! Generating predictions...")

    # Run the data through the model to get a set of predictions
    predictions = []
    for item in datas:
        mfcc = torch.from_numpy(item[0]).unsqueeze(0).unsqueeze(0)
        delta = torch.from_numpy(item[1]).unsqueeze(0).unsqueeze(0)
        delta_delta = torch.from_numpy(item[2]).unsqueeze(0).unsqueeze(0)
        prediction = model(mfcc, delta, delta_delta)
        predictions.append(prediction)

    print("Predictions generated successfully! ")
    return predictions

# Generate a subtitle file from the result produced by the model
def create_subtitle(values):
    print("Generating Subtitles... ")
    f = open("demo.srt", "w")
    for i in range(0, len(values), 1):
        seconds = i + 1 + 3
        start = '0' + str(datetime.timedelta(seconds=seconds)) + ',500'
        end = '0' + str(datetime.timedelta(seconds=seconds + 1)) + ',500'
        if seconds == 4:
            start = '0' + str(datetime.timedelta(seconds=seconds - 2)) + ',500'
        content = ''
        if values[i] < 0.5:
            content = 'hyena (' + '{:.4f}'.format(values[i].item()) + ')'
        else:
            content = 'lion (' + '{:.4f}'.format(values[i].item()) + ')'
        f.write(str(i + 1) + '\n')
        f.write(start + ' --> ' + end + '\n')
        f.write(content + '\n\n')
    print("Subtitle generated successfully! ")
    f.close()
    return True


url = "https://www.youtube.com/watch?v=nJQFQZxhpgk"
#length = process_video(url)
#move_data(length)
#convert_mfcc()
useful_data = load_mfcc()
results = apply_model(useful_data)
create_subtitle(results)
#os.system("ffmpeg -i demo.mp4 -i demo.srt -c copy -c:s mov_text demo_sub.mp4")


