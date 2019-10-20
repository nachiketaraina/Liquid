from utils1 import *
from moviepy.editor import *
import os
import shutil

path = "export.wav"

def out(path):
    chunk(path)
    lst = []
    outi = []
    for filename in os.listdir("test_recordings"):
        x = preprocess('test_recordings/' + filename)
        lst.append([x,filename])
    for i in lst:
        f = pred_speech(i[0])  # pridicting foreground or backgground
        if f==1:
            s = recognize_speech('test_recordings/' + i[1])
            outi.append(s)
        else:
            s = pred_back(i[0])  # predicting background audio type
            outi.append(s)
    
    return lst,outi

lst,outi = out(path)

lst = ["seems like this is a burger in the house", 6, 3, "be quick call the police they will be here in a moment", 8]

# os.mkdir("videoClips")
for i in range(len(lst)):
    if(type(lst[i]) == str):
        txt = TextClip(lst[i], font = 'Amiri-regular', color='white', fontsize=24)
        txt.duration = 4
        txt.write_videofile("videoClips/0" + str(i) + ".mp4", fps = 24)    
    elif(type(lst[i] == int)):
        path = "gifs/"
        dirs = os.listdir( path )
        shutil.copyfile('gifs/' + str(lst[i]) + '.mp4', 'videoClips/0' + str(i) + '.mp4')

from moviepy.editor import VideoFileClip, concatenate_videoclips

name = os.listdir("videoClips")

lsta = []
for i in name:
    clip = VideoFileClip("videoClips/"+i)
    lsta.append(clip)

final_clip = concatenate_videoclips(lsta, method='compose')
final_clip.write_videofile("my_concatenation.mp4")


from os import startfile
startfile("muxed_file.avi")

"""
import cv2
import numpy as np
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('my_concatenation.mp4')
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
 
    # Display the resulting frame
    cv2.imshow('Frame',frame)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(30) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()
"""