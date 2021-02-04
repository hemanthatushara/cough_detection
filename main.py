import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import scipy
import scipy.signal as sps
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
import pyaudio
import noisereduce as nr


model = hub.load('https://tfhub.dev/google/yamnet/1')

from os import path
from pydub import AudioSegment

# files

##Example files conversion from mp3 to wav
src = "dataset/neg-0421-083-cough-m-53.mp3"
dst = "dataset/f.wav"

# convert wav to mp3
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")


def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])

  return class_names

class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)

def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) /
                               original_sample_rate * desired_sample_rate))
    waveform = scipy.signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform


wav_file_name = 'dataset/f.wav'
sample_rate,wav_data = wavfile.read(wav_file_name, 'rb')
print(len(wav_data)/sample_rate)
#wav_data = sps.resample(wav_data, 16000)
sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

# Show some basic information about the audio.
duration = len(wav_data)/sample_rate
print(f'Sample rate: {sample_rate} Hz')
print(f'Total duration: {duration:.2f}s')
print(f'Size of the input: {len(wav_data)}')
waveform = wav_data / tf.int16.max
print(type(waveform))
scores, embeddings, spectrogram = model(waveform)
scores_np = scores.numpy()
spectrogram_np = spectrogram.numpy()
infered_class = class_names[scores_np.mean(axis=0).argmax()]
#print(class_names[scores_np.mean(axis=0).argmax()])
#print(scores_np.argsort(axis=0)[-3:][::-1])
"""if infered_class == 'Camera':
    infered_class = 'cough'
"""
print(f'The main sound is: {infered_class}')


CHUNKSIZE = 16000 # fixed chunk size
RATE = 16000

# initialize portaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

#noise window
data = stream.read(10000)
noise_sample = np.frombuffer(data, dtype=np.float32)
print("Noise Sample")
#plotAudio2(noise_sample)
loud_threshold = np.mean(np.abs(noise_sample)) * 10
print("Loud threshold", loud_threshold)
audio_buffer = []
near = 0

while(True):
    # Read chunk and load it into numpy array.
    data = stream.read(CHUNKSIZE)
    current_window = np.frombuffer(data, dtype=np.float32)

    #Reduce noise real-time
    current_window = nr.reduce_noise(audio_clip=current_window, noise_clip=noise_sample, verbose=False)

    if(audio_buffer==[]):
        audio_buffer = current_window
    else:
        if(np.mean(np.abs(current_window))<loud_threshold):
            print("Inside silence reign")
            if(near<10):
                audio_buffer = np.concatenate((audio_buffer,current_window))
                near += 1
            else:

                scores, embeddings, spectrogram = model(np.array(audio_buffer))
                scores_np = scores.numpy()
                spectrogram_np = spectrogram.numpy()
                infered_class_1 = class_names[scores_np.mean(axis=0).argmax()]
                print(infered_class_1)
                audio_buffer = []
                near
        else:
            print("Inside loud reign")
            near = 0
            audio_buffer = np.concatenate((audio_buffer,current_window))

# close stream
stream.stop_stream()
stream.close()
