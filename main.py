
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import scipy
import scipy.signal as sps
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa


model = hub.load('https://tfhub.dev/google/yamnet/1')

from os import path
from pydub import AudioSegment

# files
src = "/Users/shahn/Desktop/Nisarg/Cough_detection/clinical/original/neg/neg-0421-083-cough-m-53.mp3"
dst = "/Users/shahn/Downloads/Coswara-Data-master/Extracted_data/cough_sounds/d.wav"

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


wav_file_name = '/Users/shahn/Downloads/Coswara-Data-master/Extracted_data/cough_sounds/d.wav'
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
