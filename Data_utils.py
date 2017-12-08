import math
from os import listdir
import os
from os.path import isfile, join, splitext
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pylab as pylab


def parse_text(fileName):
    p = Path('D:\\Users\\genia\\eclipse-workspace\\Neural Comp Assignment\\ssd_keras-1-master\\data')
    pixels = []
    with open(join(p,fileName),'r') as f:
        for x in f:
            x = x.rstrip()
            if not x: continue
            #print(x)
            pixels.append(x.split(',')[1])
    values = [line.split(" ") for line in pixels]
    ii=0
    pixelMap = np.zeros((400,400,20))
    for line in values:
        person = []
        if line[1]=='0':
            a=1
        else:
            for ij in range(0,len(line)):
                if ij%2==0:
                    pixs = int(line[ij])
                    num = int(line[ij+1])
                    for jj in range(pixs,pixs+num):
                        x = math.floor(jj/400)
                        y = jj%400
                        pixelMap[x][y][ii] = 1
        ii+=1
    
    #print(pixelMap)
    return pixelMap


def remove_duplicates(k):
    new_array = [tuple(row) for row in k]
    uniques = np.unique(new_array,axis=0)
    return uniques


def bb_IoU(boxA, boxB):
# determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return interArea, boxAArea, boxBArea


def find_Corners(pixArray,fileName, slice):
    if not pixArray[:,:,slice].any():
#         print('Empty slice', filename, slice)
        return np.array(0)
    pixArraySl = np.lib.pad(pixArray[:,:,slice],1, 'constant', constant_values=0)
     
    x=[]
    y=[]
    cornerCount = 0
    cornerCoords = []
    cornerCoordsx = np.array([], dtype=np.int16).reshape(0,2)
    for xC in range(0,401):
        for yC in range(0,401):
            coord = np.array([xC,yC])
            sumCorner = pixArraySl[xC,yC]+7*pixArraySl[xC+1,yC]+5*pixArraySl[xC,yC+1]+9*pixArraySl[xC+1,yC+1]               
            if sumCorner == 1 or sumCorner==13:
                cornerCount+=1
                cornerCoordsx = np.vstack((cornerCoordsx,[xC,yC]))
                if sumCorner == 13:
                    cornerCount+=2
                    cornerCoordsx = np.vstack((cornerCoordsx,[xC+1,yC]))
                    cornerCoordsx = np.vstack((cornerCoordsx,[xC,yC+1]))
                        
            elif sumCorner==5 or sumCorner==15:
                cornerCount+=1
                cornerCoordsx = np.vstack((cornerCoordsx,[xC,yC+1]))
                if sumCorner == 15:
                    cornerCount+=2
                    cornerCoordsx = np.vstack((cornerCoordsx,[xC+1,yC]))
                    cornerCoordsx = np.vstack((cornerCoordsx,[xC,yC]))
                    
            elif sumCorner==7 or sumCorner==17:
                cornerCount+=1
                cornerCoordsx = np.vstack((cornerCoordsx,[xC+1,yC]))
                if sumCorner == 17:
                    cornerCount+=2
                    cornerCoordsx = np.vstack((cornerCoordsx,[xC+1,yC+1]))
                    cornerCoordsx = np.vstack((cornerCoordsx,[xC,yC]))

                
            elif sumCorner==9 or sumCorner==21:
                cornerCount+=1
                cornerCoordsx = np.vstack((cornerCoordsx,[xC+1,yC+1]))
                if sumCorner == 21:
                    cornerCount+=2
                    cornerCoordsx = np.vstack((cornerCoordsx,[xC+1,yC]))
                    cornerCoordsx = np.vstack((cornerCoordsx,[xC,yC+1]))
                
    
    #print(cornerCoordsx)
    hLines = []
    vLines = []
    hLinesx = np.array([], dtype=np.int16).reshape(0,4)
    vLinesx = np.array([], dtype=np.int16).reshape(0,4)
    
    for coord in cornerCoordsx:
        for coord2 in cornerCoordsx:
            if coord[0]==coord2[0] and coord[1]!=coord2[1] and coord[1]<coord2[1]:
                #hLines.append([coord[0],coord[1],coord2[0],coord2[1]])
                hLinesx = np.vstack((hLinesx,[coord[0],coord[1],coord2[0],coord2[1]]))
    for coord in cornerCoordsx:
        for coord2 in cornerCoordsx:
            if coord[1]==coord2[1] and coord[0]!=coord2[0] and coord[0]<coord2[0]:
                #vLines.append([coord[0],coord[1],coord2[0],coord2[1]])
                hLinesx = np.vstack((hLinesx,[coord[0],coord[1],coord2[0],coord2[1]]))
                
    for hLine in hLinesx:
        for vLine in vLinesx:    
            x = hLine[0]
            y = vLine[1]
            
            if x == 179 and y == 310:
#                 print('This shit right here!')
                a=1
            kernel = pixArraySl[x-1:x+2,y-1:y+2]
            if np.sum(kernel)>5:
                #print(kernel)
                cornerCount+=1
                cornerCoordsx = np.vstack((cornerCoordsx,[x,y]))
    
#     print(cornerCoordsx)
    #print(hLinesx)
    if cornerCount > 0:
        cornerCoordsx = remove_duplicates(cornerCoordsx)
    
    hLinesx = np.array([], dtype=np.int64).reshape(0,4)
    vLinesx = np.array([], dtype=np.int64).reshape(0,4)
    
    for coord in cornerCoordsx:
        for coord2 in cornerCoordsx:
            if coord[0]==coord2[0] and coord[1]!=coord2[1] and coord[1]<coord2[1]:
                #hLines.append([coord[0],coord[1],coord2[0],coord2[1]])
                hLinesx = np.vstack((hLinesx,[coord[0],coord[1],coord2[0],coord2[1]]))
    for coord in cornerCoordsx:
        for coord2 in cornerCoordsx:
            if coord[1]==coord2[1] and coord[0]!=coord2[0] and coord[0]<coord2[0]:
                #vLines.append([coord[0],coord[1],coord2[0],coord2[1]])
                hLinesx = np.vstack((hLinesx,[coord[0],coord[1],coord2[0],coord2[1]]))
                
    #print(cornerCoords)
#     print('hLinesx')
#     print(hLinesx)
    #print('vLines'),print(vLines)            
    boxes = np.array([], dtype=np.int16).reshape(0,5)
    for hLine in hLinesx:
        for vLine in hLinesx:
            #print(np.any(hLine!=vLine))
            a=1
            if hLine[1] == vLine[1] and hLine[3] == vLine[3] and np.any(hLine!=vLine) and hLine[0]<vLine[0]:
                #boxes.append([hLine[0],hLine[1],vLine[2],vLine[3],0])
                boxes = np.vstack((boxes,[hLine[0],hLine[1],vLine[2],vLine[3],0]))
    
#     print('boxes')
#     print(boxes)
    #pylab.rcParams['figure.figsize'] = (10, 10)
#     print('boxes ', len(boxes))
    #print(boxes)
    xCount = 0;

#     for x in range(0,402):
#         for y in range(0,402):
#             if pixArraySl[x,y] > 0.9:
#                 pixArraySl[x,y]=0.3
    
    boxN=0
    for box in boxes:
        any0='false'
        for x in range(box[0],box[2]):
            for y in range(box[1],box[3]):
                if pixArraySl[x,y]<0.1:
                    any0='true'
                    box[4]=1
#                     print('box has gaps ',boxN)
                    
                    #pixArraySl[x,y]+=0.2
                if any0=='true':
                    break
            if any0=='true':
                break
            
        boxN+=1
    boxN=0
    for box in boxes:
        if box[4]==0:
            v1 = np.all(pixArraySl[box[0]:box[2]+1,box[1]-1])
            v2 = np.all(pixArraySl[box[0]:box[2]+1,box[3]+1])
            v3 = np.all(pixArraySl[box[0]-1,box[1]:box[3]+1])
            v4 = np.all(pixArraySl[box[2]+1,box[1]:box[3]+1])
            if (np.all([v1,v2,v3,v4])):
                v1chk=1
                box[4]=1
#                 print('box is internal ',boxN)
            height = box[2]-box[0]
            width=box[3]-box[1]
            if width > 0 and height > 0 and (width/height>10 or height/width >10):
                v1chk=1
                box[4]=1
#                 print('aspect ratio too high ',boxN)
            boxN+=1

    boxN=0
    for box in boxes:
        for box2 in boxes:
            if np.any(box != box2) and box2[4]==0 and box[4]==0:
                iou = (bb_IoU(box,box2))
                if iou[0]==iou[1]:
#                     print(boxN,iou,box,box2)
                    v1chk=1
                    box[4]=1

    
    boxOK=0
    for box in boxes:
        if box[4]==0:
            boxOK+=1
#             print(boxOK,box[4])
#             for x in range(box[1],box[3]):
#                 pixArraySl[box[0],x] = 0.5
#                 pixArraySl[box[2],x] = 0.5
#                 a=1
#             for y in range(box[0],box[2]):
#                 pixArraySl[y,box[1]] = 0.5
#                 pixArraySl[y,box[3]] = 0.5
#                 a=1
    
    boxListOutput = np.array([], dtype=np.int16).reshape(0,6)
    for box in boxes:
        #print(box.ndim)
        if box[4]==0:
            boxListOutput = np.vstack((boxListOutput,[filename+'.jpg',slice+1,box[1],box[3],box[0],box[2]]))
    

    if boxOK>0:
        boxListOutput = remove_duplicates(boxListOutput)
        return boxListOutput
    else:
        return np.array(0)
    

def make_img(pixArray,fileName, slice):
    import matplotlib.pyplot as plt
    from PIL import Image
    import glob, os
    #plt.imshow(pixArray[:,:,slice])
    #plt.show()
    im = plt.imread(fileName)
    print(im.shape)
    im2 = im[:,:,1]*pixArray[:,:,slice]
    im4 = np.array(im)
    for i in range(3):
        im4[:,:,i] = im[:,:,i]*pixArray[:,:,slice]
    plt.imshow(im2)
    plt.savefig('current.png')
    plt.show()

    im3 = Image.fromarray(im4,mode='RGB')
    im3.save('current.jpg', "")


# 
# def make_img(pixArray,fileName, slice):
#     import matplotlib.pyplot as plt
#     from PIL import Image
#     import glob, os
#     #plt.imshow(pixArray[:,:,slice])
#     #plt.show()
#     im = plt.imread(fileName)
# #     print(im.shape)
#     im2 = im[:,:,1]*pixArray[:,:,slice]
#     plt.imshow(im2)
#     plt.savefig('current.png')
#     plt.show()
# 
#     im3 = Image.fromarray(im2,mode='L')
#     im3.save('current.bmp', "BMP")


# In[17]:

