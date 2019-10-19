import numpy as np
import pandas as pd 
from pydub import AudioSegment
import os
import librosa
from keras.models import load_model
import random
import time
import speech_recognition as sr


def chunk(path):
    audio = AudioSegment.from_wav(path)
    n = len(audio)
    counter = 1
    interval = 4 * 1000
    overlap =  0 * 1000
    start = 0
    end = 0
    flag = 0
    for i in range(0,  n, interval): 
        if(i == 0): 
            start = 0
            end = interval 
        else:
            start = end - overlap 
            end = start + interval  
        if(end >= n): 
            end = n 
            flag = 1
        chunk = audio[start:end] 
        filename = 'test_recordings/chunk' + str(counter) + '.wav'
        chunk.export(filename, format = "wav") 
        counter = counter + 1


def preprocess(path):
    x = []
    filename=path
    y,sr=librosa.load(filename)
    mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=40).T,axis=0)
    melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40,fmax=8000).T,axis=0)
    chroma_stft=np.mean(librosa.feature.chroma_stft(y=y, sr=sr,n_chroma=40).T,axis=0)
    chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr,n_chroma=40).T,axis=0)
    chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr,n_chroma=40).T,axis=0)
    features=np.reshape(np.vstack((mfccs,melspectrogram,chroma_stft,chroma_cq,chroma_cens)),(40,5))
    x.append(features)
    x = np.asarray(x)
    x = x.reshape((x.shape[0],40,5,1))
    return x

def pred_speech(x):
    model = load_model("speech_model_correct.h5")
    pred = model.predict(x)
    if pred[0,0]>pred[0,1]:
            return 0
    else:
            return 1

def pred_back(x):
    model = load_model("urbanSound.h5")
    pred = model.predict(x)
    v = np.argmax(pred,axis=1)
    return v[0]

def recognize_speech(filename):
    AUDIO_FILE = filename 
    r = sr.Recognizer() 
    with sr.AudioFile(AUDIO_FILE) as source: 
        audio_listened = r.listen(source) 
    try:     
        rec = r.recognize_google(audio_listened) 
        return rec
    except sr.UnknownValueError: 
        return " "
    except sr.RequestError as e: 
        return " " 
  
 
