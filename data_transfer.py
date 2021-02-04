import os
import shutil
import glob


z = 1
import os
cwd = os.getcwd()
"""FINAL_DIR = cwd+'/cough_sounds/'

#FINAL_DIR = cwd+''
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith("cough-heavy.wav") or file.endswith("cough-shallow.wav") :
             os.rename(os.path.join(root, file), FINAL_DIR + str(z) + ".wav"  )
             z +=1

print(z)
"""
FINAL_DIR_NOISE = cwd+'/white_noise/'

for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith("counting_normal.wav") or file.endswith("counting_fast.wav") or file.endswith("vowel-o.wav") or file.endswith("vowel-a.wav") or file.endswith("vowel-e.wav"):
             if z < 2972:
                 os.rename(os.path.join(root, file), FINAL_DIR_NOISE + str(z) + ".wav"  )
                 z +=1

print(z)
