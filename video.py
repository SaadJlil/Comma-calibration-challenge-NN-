import cv2 
import numpy as np
from glob import glob
import time

#convert video .hevc to numpy arrays 

for a in glob("*.hevc"):
    cap = cv2.VideoCapture(a)
    ret = True 
    frames = []
    i = 0
    b = 0
    while ret:
        ret, img = cap.read()
        if ret:
            frames.append(img)
        if i == 200 or (a=="4.hevc" and b == 5 and i == 196):
            i = 0
            frames = np.stack(frames, axis = 0)
            np.save("./footage_np/footage{}.{}".format(a[:-5],b), frames) 
            
            frames = []
            b += 1
        if (b == 6):
            break
        i += 1

# results from txt to numpy

"""
b = 0

for a in glob("*.txt"):
    file = open(a, "r")
    other = list()

    txt = file.readlines()


    for i in txt:
        other.append(i.split(" "))

    for i in range(len(other)):
        for j in range(2):
            other[i][j] = other[i][j].replace('\n', '')


    other = np.array(other)
    for j,i in enumerate(other):
        while (other[j+b] == np.array(['nan', 'nan'])).any():
            b += 1
        if b > 0:
            for z in range(b):
                other[j+z] = other[j+z-1].astype(float) + (other[j+b].astype(float) - other[j-1].astype(float))/float(b)
            b = 0

    

    file.close()
"""



"""
b = 0
data = None

for a in glob("*.txt"):
    file = open(a, "r")
    other = list()

    txt = file.readlines()
    
    for i in txt:
        other.append(i.split(" "))

    for i in range(len(other)):
        for j in range(2):
            other[i][j] = other[i][j].replace('\n', '')


    other = np.array(other)
    b = 0
    for j,i in enumerate(other):
        while (other[j+b] == np.array(['nan', 'nan'])).any() and (b+j) < other.shape[0]-1:
            b += 1
        if b > 0:
            for z in range(b):
                if (other[j+b] == np.array(['nan', 'nan'])).any(): 
                    other[j+z] = other[j-1]
                    if z == b-1:
                        other[j+b] = other[j-1]
                else:
                    other[j+z] = other[j+z-1].astype(float) + (other[j+b].astype(float) - other[j-1].astype(float))/float(b)
            b = 0


    if a == "3.txt":
        data = other
    else:
        data = np.concatenate([data, other])

    file.close()

"""
