'''
Data Collection for FieldworkAnimalDetector, the program only works under Ken's machine and set up.
'''

import os
import random

# Download the source audio from a set of urls
# Requires the installation of youtube-dl and ffmpeg in the local machine
# Assumes the softwares are under the same folder as this program

os.system('"youtube-dl -x --audio-format wav --output "hyena_sample.%(ext)s" https://www.youtube.com/watch?v=k1kVc4qsspY"')
os.system('"youtube-dl -x --audio-format wav --output "lion_sample.%(ext)s" https://www.youtube.com/watch?v=9mizXaQYUd0"')

# Create data samples by trimming the large audio files at random time stamps

# An upper bound of time stamp, determined by experiment
max_t = 21600

for i in range(1, 501, 1):
    print("Sample # " + str(i))
    t_hyena = float(random.randint(10, max_t))
    t_lion = float(random.randint(10, max_t))
    command_hyena = "ffmpeg -i hyena_sample.wav -ss " + str(t_hyena) + " -to " + str(
        t_hyena + 1.5) + " -c copy hyena" + str(i) + ".wav"
    command_lion = "ffmpeg -i lion_sample.wav -ss " + str(t_lion) + " -to " + str(
        t_lion + 1.5) + " -c copy lion" + str(i) + ".wav"
    os.system(command_hyena)
    os.system(command_lion)

print("Processing completed! ")