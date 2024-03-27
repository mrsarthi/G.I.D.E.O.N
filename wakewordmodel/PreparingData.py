import sounddevice as sd
from scipy.io.wavfile import write
# sounddevice is used to record the sounds and make a numpy array of it
# # the above import takes the numpy array and saves it a wav file


def record_audio_and_save(save_path, n_times=550):
    # this function will record the sounds that contain wake word
    # save_path is a path to an empty directory where it should save all the audio files

    input("To start recording Wake Word press Enter: ")
    for i in range(n_times):
        fs = 44100  # number of samples in a single sound (it's like fps)
        seconds = 2

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        # seconds * fs means number of samples in a recorded sound
        # channel means dual channel audio and not mono channel

        sd.wait()  # this function will allow the program to wait until the recording is done

        write(save_path + str(i) + ".wav", fs, myrecording)
        input(
            f"Press to record next or two stop press ctrl + C ({i + 1}/{n_times}): ")


def record_background_sound(save_path, n_times=500):
    # this function will record the background sounds instead of wake word

    input("To start recording your background sounds press Enter: ")
    for i in range(n_times):
        fs = 44100
        seconds = 3

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()

        write(save_path + str(i) + ".wav", fs, myrecording)

        print(f"Currently on {i+1}/{n_times}")


# Step 1: Record yourself saying the Wake Word
print("Recording the Wake Word:\n")
record_audio_and_save("audio_data/", n_times=550)

# Step 2: Record your background sounds (Just let it run, it will automatically record)
# print("Recording the Background sounds:\n")
# record_background_sound("background_sound/", n_times=500)
