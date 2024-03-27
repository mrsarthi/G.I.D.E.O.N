import webbrowser
import os
from playsound import playsound
from gtts import gTTS
import speech_recognition
import json


def tts(voice_inp_to_Tex):

    text_val = voice_inp_to_Tex
    language = 'en'
    obj = gTTS(text=text_val, lang=language, slow=False)
    obj.save("response.mp3")
    playsound("response.mp3")


inp_voice_rec = speech_recognition.Recognizer()
print('Gideon is up, what do you want me to do...')
# playsound("Hello_There_Obi_Wan_-_Sound_Effect_HD.mp3")
while (1):
    try:
        with speech_recognition.Microphone() as inp_cource:

            inp_voice_rec.adjust_for_ambient_noise(
                inp_cource, duration=0.5)

            voice_inp = inp_voice_rec.listen(inp_cource)

            voice_inp_to_Text = inp_voice_rec.recognize_google(
                voice_inp)
            voice_inp_to_Text = voice_inp_to_Text.lower()
            # print(voice_inp_to_Text)
            if voice_inp_to_Text == 'exit' or voice_inp_to_Text == 'quit':
                tts('ok bye')
                exit(0)
            if voice_inp_to_Text == 'open youtube':
                url = "https://www.youtube.com/"
                webbrowser.open(url)
                tts('Opening youtube')
            if voice_inp_to_Text == 'open github':
                url = "https://www.github.com/mrsarthi"
                webbrowser.open(url)
                tts('Opening github')

            # elif voice_inp_to_Text == "ronaldo" or "su":
            #     playsound(r"C:\Users\wfors\Downloads\siuuu.mpeg")
                # print(voice_inp_to_Text)
            elif voice_inp_to_Text == 'hello' or 'hai' or 'konichiwa':
                tts('Hai there, how can I help you')
            else:
                tts('repeat please')

    except KeyboardInterrupt:
        print('A KeyboardInterrupt encountered; Terminating the Program !!!')
        exit(0)

    except speech_recognition.UnknownValueError:
        print("Background sound only detected, no conversion")
