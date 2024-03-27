import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


walley_sample = "background_sound/420.wav"
data, sample_rate = librosa.load(walley_sample)
# librosa.load takes the path of an audio file and returns the numpy array and the sample rate of the audio


##### VISUALIZING WAVE FORM ##
# Plotting the sound using matplotlib
plt.title("Wave Form")
librosa.display.waveshow(data, sr=sample_rate)
plt.show()

# Mel-Frequency Cepstral Coefficients
# the Mel Frequency Cepstral Transform can be used to calculate the short-term power spectrum of a sound in the form of Cepstral coefficients
# The power spectrum of a signal indicates the relative magnitudes of the frequency components that combine to make up the signal
mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
print("Shape of mfcc:", mfccs.shape)

plt.title("MFCC")
librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
plt.show()

##### Doing this for every sample ##

all_data = []

data_path_dict = {
    # class level 0 and 1 (0 for no, 1 for yes)
    0: ["background_sound/" + file_path for file_path in os.listdir("background_sound/")],
    1: ["audio_data/" + file_path for file_path in os.listdir("audio_data/")]
}

# the background_sound/ directory has all sounds which DOES NOT CONTAIN wake word
# the audio_data/ directory has all sound WHICH HAS Wake word

for class_label, list_of_files in data_path_dict.items():
    for single_file in list_of_files:
        audio, sample_rate = librosa.load(single_file)  # Loading file
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sample_rate, n_mfcc=40)  # Apllying mfcc
        mfcc_processed = np.mean(mfcc.T, axis=0)  # some pre-processing
        all_data.append([mfcc_processed, class_label])
    print(f"Info: Succesfully Preprocessed Class Label {class_label}")

df = pd.DataFrame(all_data, columns=["feature", "class_label"])

###### SAVING FOR FUTURE USE ###
df.to_pickle("final_audio_data_csv/audio_data.csv")
