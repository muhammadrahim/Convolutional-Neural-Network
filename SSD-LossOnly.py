
#get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import timedelta
import math
from os import listdir
import os
from os.path import isfile, join, splitext
from pathlib import Path
import time
from scipy import sparse, savez
from sklearn.metrics import confusion_matrix
from sys import getsizeof
import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import pylab as pylab
import Data_utils

os.chdir('C:\\Users\\stpen\\eclipse-workspace\\Master SSD\\')
p = Path('C:\\Users\\stpen\\eclipse-workspace\\Master SSD\\data')

def parse_pixel_map(pixelMap): #in 1
    priority = [0 , 1, 10, 19, 2, 20, 3, 6, 9, 8, 13, 12, 16, 14, 15, 5, 18, 11, 7, 4, 17]
    pixelMap = pixelMap.astype(np.uint8)
    reducedPM = np.ones([400,400], dtype=np.bool) # 1
    newMap = np.zeros([400,400,21], dtype=np.uint8)
    for ii in range(20):
        slice = pixelMap[:,:,ii]
        if slice.any():
            pixelMap[:,:,ii]=slice*(priority[ii+1]+1) 
    
    #print(pixelMap)
    for ii in range(21):
        a=1

    pixelMap = np.pad(pixelMap, ((0, 0),(0,0),(1,0)), 'constant', constant_values=1)
    reducedPM = np.amax(pixelMap,axis=2, keepdims=1)
    
    for ii in range(21):
        newMap[:,:,ii:ii+1] = np.where(reducedPM[:,:,0:1]==pixelMap[:,:,ii:ii+1],1,0)
    
    newMap2 = newMap[0:400:8,0:400:8,:]
    newMap2 = np.pad(newMap2, ((0, 1),(0,1),(0,0)), 'edge')
    return newMap2

''' NOTE(GP) This method parses the original image_name.txt file and returns a 400*400*20 array, which it passes to 
parse_pixel_map automatically. 
Call this method to obtain a new "ground truth" array from an image filename.
'''
def parse_text(fileName):
    pixels = []
    with open(join(p,fileName+'.txt'),'r') as f:
        for x in f:
            x = x.rstrip()
            if not x: continue
            #print(x)
            pixels.append(x.split(',')[1])
    values = [line.split(" ") for line in pixels]
    ii=0
    pixelMap = np.zeros((400,400,20), dtype=np.bool)
    for ii,line in enumerate(values):
        person = []
        if line[1]!='0':
            for ij in range(0,len(line)):
                if ij%2==0:
                    pixs = int(line[ij])
                    num = int(line[ij+1])
                    for jj in range(pixs,pixs+num):
                        x = math.floor(jj/400)
                        y = jj%400
                        pixelMap[x][y][ii] = 1
        ii+=1
    
    newMap = parse_pixel_map(pixelMap)
        
    #print(pixelMap)
    return newMap


os.chdir('C:\\Users\\stpen\\eclipse-workspace\\Master SSD\\')
p = Path('C:\\Users\\stpen\\eclipse-workspace\\Master SSD\\data')
onlyfilesTxt = [f for f in listdir(p) if f.endswith('.txt')]
onlyfileNames = [splitext(f)[0] for f in onlyfilesTxt ]


pcsv = Path('C:\\Users\\stpen\\eclipse-workspace\\Master SSD\\data\\')
# start=0
# stop=14625
# #with open(join(pcsv,'labelsNewTest.npy'),'ab') as f:
# print('File open')
# length = stop-start
# store = np.zeros((length,2601), dtype=np.uint16)
# for ii,filename in enumerate(onlyfileNames[start:stop]):
#     exp_output = parse_text(onlyfileNames[ii])
#     exp_reshaped = np.reshape(exp_output, 54621)
#     exp_indeces = np.nonzero(exp_reshaped)
#     #np.savetxt(f, exp_indeces, delimiter=',', fmt='%i', newline='\n')    
#     store[ii:ii+1,0:2601] = exp_indeces
#     if ii%100==0:
#         print(ii)
# np.save('labelsNewTest.npy', store)
# np.savetxt('labelsNewTest.csv', store, delimiter=',', fmt='%i', newline='\n')

import timeit

data = np.load('.\LabelsMaster.npy')
batch_size = 64
data2 = data[10:10+batch_size,:]
y_truth_zeros = np.zeros((batch_size,51*51*21,1),dtype=np.bool)
for i in range(batch_size):
    "y_truth_zeros"[i:i+1,data2[i:i+1,:],0] = 1

"data3" = np.reshape(y_truth_zeros,(batch_size,51,51,21))
"data4" = np.delete(data3,3,axis=0)


def toTime():
    with open(join(pcsv,'labelsNewTest.csv'),'ab') as f:
        print('File open')
        for ii,filename in enumerate(onlyfileNames[1:2]):
            exp_output = parse_text(onlyfileNames[i])
            exp_reshaped = np.reshape(exp_output, 54621)
            exp_indeces = np.nonzero(exp_reshaped)
            np.savetxt(f, exp_indeces, delimiter=',', fmt='%i', newline='\n') 
            
    #data = np.genfromtxt(join(pcsv,'labelsNew.csv'),delimiter=',', dtype=np.uint16)
    #np.ones(2000)*np.eye(2000)
    
print(getsizeof(data), print(data.shape))



if __name__ == '__main__':
    import timeit
    print(timeit.Timer('exp_output = parse_text(onlyfileNames[i])', setup='from __main__ import parse_text').timeit(number=1))


fig, axes = plt.subplots(3, 7)
dataslice = data3[60,:,:,:]
for i in range(3):
    for j in range(7):
        index = (((i)*7)+(j))
        axes[i, j].imshow(dataslice[:,:,index])
        plt.subplots_adjust(wspace=0, hspace=0)
plt.show()
