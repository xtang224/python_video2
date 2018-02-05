#!/usr/bin/env python

'''
face detection using haar cascades

USAGE:
    python faceTrain.py 
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import io
import math

# local modules
#from video import create_capture
#from common import clock, draw_str


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def resize(img):
    mm = 9
    nn = 9
    v = img.shape
    rows = v[0]
    cols = v[1]
    img2 = np.zeros((9,9),np.float)
    rowRatio = rows / 9
    colRatio = cols / 9
    for i in range(rows):
       for j in range(cols):
        
          newRow = int(i / rowRatio)
          if newRow>8:
             newRow = 8          
          newCol = int(j / colRatio)
          if newCol > 8: 
             newCol = 8
          newAve = (img[i,j][0]*0.3 + img[i,j][1]*0.3 + img[i,j][2]*0.3)
          img2[newRow, newCol] = img2[newRow, newCol] + newAve
    for i in range(mm):
       for j in range(nn):
          img2[i,j] = img2[i,j] / (rows/mm * cols/nn)
    print ("img2.szie=", img2.size)
    return img2

def gray(img):
   tmpImg = resize(img)
   #img3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   img3 = tmpImg
   sum = 0
   v = img3.shape
   rows = v[0]
   cols = v[1]
   for i in range(rows):
       for j in range(cols):
          sum += img3[i,j]
   avg = sum / (rows*cols)
   for i in range(rows):
       for j in range(cols):
          if img3[i,j]>avg:
             img3[i,j] = 1
          else:
             img3[i,j] = 0
   print("img3.size =", img3.size)
   img3 = img3.reshape(1,img3.size)

   result = ""  
   for i in range(img3.size):
      result = result + str(int(img3[0,i]))      
   print (result)
   f = open("temp", "w")
   f.write(result)
   f.close()
   return img3

def judge(imgResult):
    passValue = 0
    for i in range(3):
       dist = 0
       target = reduce[i]
       for j in range(81):
          if (imgResult[j]!=target[j]):
             dist = dist + 1
       if dist<=30:
          passValue = passValue + weight[i] * 1
       else:
          passValue = passValue - weight[i] * 1
    if passValue>0:
       return 1
    else:
       return -1
       

if __name__ == '__main__':
    import sys, getopt
    print(__doc__)   

    threshold = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    reduce = ["111101111111000011111000011111000011111000011111000011111000011111000011111000111", "110000000110100001100100001100110001100111001100111101100111111100111111110111111", "111111111111111111100000001000000000000000000000000000100000001111111111111111111"]
    reduce2 = ["010000000111000000010000000111000000010000000111000000010000000000000000000000000", "000000000010000000111000000010000000111000000010000000111000000010000000000000000", "000000000000000000010000000111000000010000000111000000010000000111000000010000000", "001000000011100000001000000011100000001000000011100000001000000000000000000000000", "000000000001000000011100000001000000011100000001000000011100000001000000000000000", "000000000000000000001000000011100000001000000011100000001000000011100000001000000", "000100000001110000000100000001110000000100000001110000000100000000000000000000000", "000000000000100000001110000000100000001110000000100000001110000000100000000000000", "000000000000000000000100000001110000000100000001110000000100000001110000000100000", "000010000000111000000010000000111000000010000000111000000010000000000000000000000", "000000000000010000000111000000010000000111000000010000000111000000010000000000000", "000000000000000000000010000000111000000010000000111000000010000000111000000010000", "000001000000011100000001000000011100000001000000011100000001000000000000000000000", "000000000000001000000011100000001000000011100000001000000011100000001000000000000", "000000000000000000000001000000011100000001000000011100000001000000011100000001000", "000000100000001110000000100000001110000000100000001110000000100000000000000000000", "000000000000000100000001110000000100000001110000000100000001110000000100000000000", "000000000000000000000000100000001110000000100000001110000000100000001110000000100", "000000010000000111000000010000000111000000010000000111000000010000000000000000000", "000000000000000010000000111000000010000000111000000010000000111000000010000000000", "000000000000000000000000010000000111000000010000000111000000010000000111000000010"]
    error = [0, 0 ,0, 0, 0, 0, 0 ,0, 0, 0, 0, 0 ,0, 0, 0, 0, 0 ,0, 0, 0, 0]
    weight = [0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0 ,0, 0, 0, 0, 0 ,0, 0, 0, 0]
    ratio = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    origJudge = [1,1,1,1,1,-1,-1,-1,-1,-1]
    testResults = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
    imgResults = ["", "", "", "", "", "", "", "", "", ""]
    imgTestResults = ["", "", "", "", ""]
    
    round = 1
    while (round<=3):
       error = [0, 0 ,0, 0, 0, 0, 0 ,0, 0, 0, 0, 0 ,0, 0, 0, 0, 0 ,0, 0, 0, 0]
       testResults = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
       name = "redLightTrain"
       for i in range(5):
          name = "redLightTrain"
          name = name + str(i+1) + ".jpg"                 
          if round==1:
             img =  cv2.imread(name);
             tImg = gray(img)
             
             f = open("temp")
             line = f.readline()
             f.close()
             imgResults[i] = line
             
          for count in range(3):
             target = reduce[count]
             dist = 0
             for j in range(81):   
                #print("j=", str(j), " and reduce[count][j]=", reduce[count][j], " and imgResults[i][j]=", imgResults[i][j])          
                if (reduce[count][j]!=imgResults[i][j]):
                   dist = dist + 1
             print("the ", str(i), "th picture is true, and with weak classifier ", str(count), " and the distance is ", str(dist))
             if dist>30:
                testResults[i][count] = -1
                judge = testResults[i][count] * origJudge[i]
                if judge==1:
                   print("the ", str(i), "th picture is false, and get false with weak classifier ", str(count), " and the distance is ",  str(dist))
                else:
                   error[count] = error[count] + 1
                   print("the ", str(i), "th picture is true, but get false with weak classifier ", str(count), " and the distance is ",  str(dist))               
             else:
                testResults[i][count] = 1
                judge = testResults[i][count] * origJudge[i]
                if judge==1:
                   print("the ", str(i), "th picture is true, and get true with weak classifier ", str(count), " and the distance is ",  str(dist))
                else:
                   error[count] = error[count] + 1
                   print("the ", str(i), "th picture is false, but get true with weak classifier ", str(count), " and the distance is ",  str(dist))     

       name = "redLightTrainN"
       for i in range(5):
          name = "redLightTrainN"
          name = name + str(i+1) + ".jpg"          
          #cv2.imshow("this", img)
          #cv2.waitKey(0);
          if round==1:
             img =  cv2.imread(name);
             tImg = gray(img)
            
             f = open("temp")
             line = f.readline()
             f.close()
             imgResults[i+5] = line
             
          for count in range(3):
             target = reduce[count]
             dist = 0
             for j in range(81):             
                if (target[j]!=imgResults[i+5][j]):
                   dist = dist + 1
             print("the ", str(i+5), "th picture is false, and with weak classifier ", str(count), " and the distance is ", str(dist))
             if dist>30:
                testResults[i+5][count] = -1
                judge = testResults[i+5][count] * origJudge[i+5]
                if judge==1:
                   print("the ", str(i+5), "th picture is false, and get false with weak classifier ", str(count), " and the distance is " + str(dist))
                else:
                   print("the ", str(i+5), "th picture is true, but get false with weak classifier ", str(count), " and the distance is " + str(dist))
                   error[count] = error[count] + 1 
             else:
                testResults[i+5][count] = 1
                judge = testResults[i+5][count] * origJudge[i+5]
                if judge==1:
                   print("the ", str(i+5), "th picture is true, and get true with weak classifier ", str(count), " and the distance is " + str(dist))
                else:
                   print("the ", str(i+5), "th picture is false, but get true with weak classifier ", str(count), " and the distance is " + str(dist))
                   error[count] = error[count] + 1
       minError = 20
       minUse = -1
       for i in range(3):
          if error[i]<minError:
             minError = error[i]
             minUse = i
       print("at the end of the first round, ,minError=", str(minError), " and minUse =", str(minUse))

       errorRatio = float(minError)/10
       alpha = 1
       try:
          alpha = 0.5 * math.log((1-errorRatio)/errorRatio)
       except Exception, e:
          print ("exception",":",e)
          print("errorRatio for classifier ", str(minError), " is zero. Therefore, we will stop here and use ", str(minUse), " as the strong classifier")       
       weight[minUse] = alpha
       if alpha == 1:
          break
 
       #adjust ratio for these training targets
       totalRatio = 0
       for i in range(5):
          totalRatio = totalRatio + ratio[i] * math.exp(-alpha * origJudge[i] * testResults[i][minUse])
       for i in range(5):
          totalRatio = totalRatio + ratio[i+5] * math.exp(-alpha * origJudge[i+5] * testResults[i+5][minUse])
       for i in range(5):
          ratio[i] = ratio[i] * math.exp(-alpha * origJudge[i] * testResults[i][minUse]) / totalRatio
       for i in range(5):
          ratio[5+i] = ratio[5+i] * math.exp(-alpha * origJudge[5+i] * testResults[5+i][minUse]) / totalRatio
       print("in the round of ", str(round), str(minUse), "th weak classifier performs the best and the weight is ", str(alpha))
       
       round = round + 1
  
    #now we are going to check the new picture with our strong classifier
    for i in range(5):
       name = "redLightTest"
       name = name + str(i+1) + ".jpg"   
       img =  cv2.imread(name);
       tImg = gray(img)
            
       f = open("temp")
       line = f.readline()
       f.close()
       imgTestResults[i] = line 
       #testResult = judge(imgTestResults[i])

       passValue = 0
       for i2 in range(3):
          dist = 0
          target = reduce[i2]
          for j in range(81):
             if (imgTestResults[i][j]!=target[j]):
                dist = dist + 1
          if dist<=30:
             passValue = passValue + weight[i2] * 1
          else:
             passValue = passValue - weight[i2] * 1
       if passValue>0:
          print("for the ", str(i), " th test as true value, it is judged as (1 for true and -1 for false): ", str(1))
       else:
          print("for the ", str(i), " th test as true value, it is judged as (1 for true and -1 for false): ", str(-1))
       
    
