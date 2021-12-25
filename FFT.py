#!/usr/bin/env python
# coding: utf-8

# In[44]:


# import library
import numpy as np
import cv2
import pandas as pd
import itertools
import os
import copy


# In[38]:


# for counting time
import time

def tic():
    #require  to import time
    global start_time_tictoc
    start_time_tictoc = time.time()


def toc(tag="elapsed time"):
    if "start_time_tictoc" in globals():
        print("{}: {:.9f} [sec]".format(tag, time.time() - start_time_tictoc))
    else:
        print("tic has not been called")


# In[39]:


### version1 addframe, FFT### the case: total grid size is odd (e.g. 5X5, 21X21)
def addframe(num, grayvalue, img): # for FFT calculation, 
    # add pixel
    num_insert = int((num-1)/2)
    # create block
    bk_tb = np.zeros((num_insert,img.shape[1]),np.uint8)
    bk_lr = np.zeros((img.shape[0]+(num_insert*2),num_insert),np.uint8)
    # decide gray value
    bk_tb[:,:] = grayvalue # white
    bk_lr[:,:] = grayvalue # white
    # insert pixel (top, bottom)
    array = np.insert(img, 0, bk_tb, axis=0)
    array = np.insert(array, array.shape[0], bk_tb, axis=0)

    # insert pixel (left, right)
    array = np.insert(array, [0], bk_lr, axis=1)
    array = np.insert(array, [array.shape[1]], bk_lr, axis=1)
    
    return np.array(array)

def FFT(img, num):
    amp = []
    phase = []
    num_insert = int((num-1)/2)
    h = img.shape[0] - (num_insert*2)
    w = img.shape[1] - (num_insert*2)
    for  y in range(num_insert,img.shape[0]-num_insert):
        for x in range(num_insert,img.shape[1]-num_insert):
            img1 = img[(y-num_insert):(y+num_insert+1),(x-num_insert):(x+num_insert+1)]
            fimg = np.fft.fft2(img1)
            a = np.abs(fimg)
            p = np.angle(fimg)
            amp.append(a[num_insert, num_insert]) # centre coordinates 
                                                  # (e.g. 5X5([0,0]~[4,4]) ->[2,2])
            phase.append(p[num_insert, num_insert]) # centre coordinates
                                                    # (e.g. 5X5([0,0]~[4,4]) ->[2,2])
    amp = np.reshape(amp,(h,w))
    phase = np.reshape(phase,(h,w))
    return amp, phase
#### end ################################################################################

### update addframe, FFT### the case: the number of grid is even, merge 4pixels into one
def addframe_update(num, grayvalue, img): # for FFT calculation in case of NXN
    # add pixel
    num_insert = int(num/2) 
    # create block (top:num_insert, bottom:num_insert+1, left:num_insert, right:num_insert+1)
    bk_t = np.zeros((num_insert-1,img.shape[1]),np.uint8)
    bk_b = np.zeros((num_insert,img.shape[1]),np.uint8)
    bk_l = np.zeros((img.shape[0]+(num_insert*2)-1,num_insert-1),np.uint8)
    bk_r = np.zeros((img.shape[0]+(num_insert*2)-1,num_insert),np.uint8)
    # decide gray value
    bk_t[:,:] = grayvalue # white
    bk_b[:,:] = grayvalue # white
    bk_l[:,:] = grayvalue # white
    bk_r[:,:] = grayvalue # white
    # insert pixel (top, bottom)
    array = np.insert(img, 0, bk_t, axis=0)
    array = np.insert(array, array.shape[0], bk_b, axis=0)

    # insert pixel (left, right)
    array = np.insert(array, [0], bk_l, axis=1)
    array = np.insert(array, [array.shape[1]], bk_r, axis=1)
    
    return np.array(array)


def FFT_update(img, num):
#Note num = more than 4 and total grid size is even
    amp_4 =[]
    phase_4 = []
    num_insert = int(num/2)
    h = img.shape[0] - (num_insert*2)+1
    w = img.shape[1] - (num_insert*2)+1
    for  y in range(num_insert-1,img.shape[0]-num_insert):
        for x in range(num_insert-1,img.shape[1]-num_insert):
            img1 = img[(y-num_insert+1):(y+num_insert+1),(x-num_insert+1):(x+num_insert+1)]
            h1,w1 = img1.shape[0],img1.shape[1]
            img_comp = []
#### for 1/4 compression e.g. 32->8, 12->3 ####
#             for i in range(0,h1,4):
#                 for j in range(0,w1,4):
#                     img_comp.append(np.average(img1[i:i+4,j:j+4]))
#             img_comp = np.reshape(img_comp,(int(h1/4),int(w1/4)))
###############################################

#### for 1/2 compression e.g. 32->16, 12->6 6->3 ####
            for i in range(0,h1,2):
                for j in range(0,w1,2):
                    img_comp.append(np.average(img1[i:i+2,j:j+2]))
            img_comp = np.reshape(img_comp,(int(h1/2),int(w1/2)))
#####################################################

#### for 1/3 compression e.g. 6->2 ####
#             for i in range(0,h1,3):
#                 for j in range(0,w1,3):
#                     img_comp.append(np.average(img1[i:i+3,j:j+3]))
#             img_comp = np.reshape(img_comp,(int(h1/3),int(w1/3)))
#######################################

            fimg = np.fft.fft2(img_comp)
            a = np.abs(fimg) # amplitude
            p = np.angle(fimg) # phase

#### for 1/2 compression e.g. 32->16, 12->6 6->3
            if (h1%4 !=0) and (w1%4 != 0):
                amp_4.append(a[int(h1/4),int(w1/4)])
                phase_4.append(a[int(h1/4),int(w1/4)])
            else:
                amp_4.append(np.average(a[int(h1/4)-1:int(h1/4)+1,int(w1/4)-1:int(w1/4)+1]))
                phase_4.append(np.average(p[int(h1/4)-1:int(h1/4)+1,int(w1/4)-1:int(w1/4)+1]))
################################################

#### for 1/3 compression e.g. 6->2
#             amp_4.append(np.average(a[0:2,0:2]))
#             phase_4.append(np.average(p[0:2,0:2]))
##################################

    amp_4 = np.reshape(amp_4,(h,w))
    phase_4 = np.reshape(phase_4,(h,w))
    return amp_4, phase_4
###### end ##################################################################################

#### add label to separate the pixels on the edge for detecting edge ####
# position 0: white 1: part of shading and edge 2: part of shading and not edge 
#                   3: part of symbol and edge  4: part of symbol and not edge 
def detect_edge(array,shading,txt,txt_ad):
    edge =[]
    num = 1
    for y in range(num,array.shape[0]-num):
        for x in range(num,array.shape[1]-num):
            if array[y,x] == 255:
                edge.append(0)
            elif (array[y,x] == 0) and (array[y,x] == shading[y-num,x-num]) and             (array[y,x] != txt[y-num,x-num]):
                total_count = 0
                for i,j in itertools.product(range(-1,2),range(-1,2)):
                    if array[y+i,x+j] == 255:
                        total_count += 1
                if total_count >= 1:
                    edge.append(1)
                else:
                    edge.append(2)
            elif (array[y,x] == 0) and (array[y,x] != shading[y-num,x-num]) and             (array[y,x] == txt[y-num,x-num]):
                total_count1 = 0
                for k,l in itertools.product(range(-1,2),range(-1,2)):
                    if txt_ad[y+k,x+l] == 255:
                        total_count1 += 1
                if total_count1 >= 1:
                    edge.append(3)
                else:
                    edge.append(4)
            else:
                total_count2 = 0
                for m,n in itertools.product(range(-1,2),range(-1,2)):
                    if txt_ad[y+m,x+n] == 255:
#                     if array[y+m,x+n] == 255:
                        total_count2 += 1
                if total_count2 >= 1:
                    edge.append(3)
                else:
                    edge.append(4)
    return edge
#### end #################################################################


# In[43]:


tic()

# assign the area applying FFT
# num = 31 # ver.1 num * num (e.g. 31X31(15+1+15)) 
num = 6 # num * num (e.g. 32X32(15+2+15))
# file name pattern: 1.total size is odd -> e.g. output5_csv
#                    2.total size is even and merge 4pixels into one -> e.g. output6_3_csv
#                    3.add feature for separating pixels on the edge -> e.g. output5_de_csv
filename = str(num)+'_3_csv'
os.makedirs('./step2/output'+filename,exist_ok=True) # create directory
number_of_data = 200 # the number of creating artificial images
for i in range(number_of_data):
    # load image
    img_shade = cv2.imread('./test/output/data'+str(i+1)+'shading.png')
    img_text = cv2.imread('./test/output/data'+str(i+1)+'text.png')
    img_combined = cv2.imread('./test/output/data'+str(i+1)+'combined.png')

    # get information per pixel 
    gray_shade = cv2.cvtColor(img_shade,cv2.COLOR_RGB2GRAY)
    gray_text = cv2.cvtColor(img_text,cv2.COLOR_RGB2GRAY)
    gray_combined = cv2.cvtColor(img_combined,cv2.COLOR_RGB2GRAY)

    # add frame for FFT (the case:total size is odd)
#     gray_combined_ex = addframe(num,255,gray_combined)
#     amp,phase = FFT(gray_combined_ex, num)

    # Update add frame for FFT(the case:total size is even)
    gray_combined_ex = addframe_update(num,255,gray_combined)
    amp_4,phase_4 = FFT_update(gray_combined_ex, num)
    
    # add frame for distance to the edge ## update add predictors
#     gray_combined_ad = addframe(ck_gv_grid,255,gray_combined)
#     gray_text_ad = addframe(ck_gv_grid,255,gray_text)
#     edge = detect_edge(gray_combined_ad, gray_shade, gray_text, gray_text_ad)
#     edge = np.reshape(edge, (gray_combined.shape[0],gray_combined.shape[1]))

    # get image size
    height, width = gray_combined.shape[:2]

    # initialise
    data =[]

### PGM update add FFT value, distance to the edge as predictors ###
# label 0: white 1: part of shading 2: part of symbol
# pattern 1: the grid size is odd, 2: the grid size is even,
#         3: add feature regarding pixels on the edge
    for y in range(height):
        for x in range(width):
            if gray_combined[y,x] == 255:
#                 data.append([y,x,gray_combined[y,x],amp[y,x],phase[y,x]],0) # pattern1
                data.append([y,x,gray_combined[y,x],amp_4[y,x],phase_4[y,x],0]) # pattern2
#                 data.append([y,x,gray_combined[y,x],amp[y,x],phase[y,x],edge[y,x],0]) # pattern3
            elif (gray_combined[y,x] == 0) and (gray_combined[y,x] == gray_shade[y,x]) and                  (gray_combined[y,x] != gray_text[y,x]):
#                 data.append([y,x,gray_combined[y,x],amp[y,x],phase[y,x]],0) # pattern1 
                data.append([y,x,gray_combined[y,x],amp_4[y,x],phase_4[y,x],1]) # pattern2
#                 data.append([y,x,gray_combined[y,x],amp[y,x],phase[y,x],edge[y,x],1]) # pattern3
            elif (gray_combined[y,x] == 0) and (gray_combined[y,x] != gray_shade[y,x]) and                  (gray_combined[y,x] == gray_text[y,x]):
#                 data.append([y,x,gray_combined[y,x],amp[y,x],phase[y,x]],0) # pattern1
                data.append([y,x,gray_combined[y,x],amp_4[y,x],phase_4[y,x],2]) # pattern2
#                 data.append([y,x,gray_combined[y,x],amp[y,x],phase[y,x],edge[y,x],2]) # pattern3
            else:
#                 data.append([y,x,gray_combined[y,x],amp[y,x],phase[y,x]],0) # pattern1
                data.append([y,x,gray_combined[y,x],amp_4[y,x],phase_4[y,x],1]) # pattern2
#                 data.append([y,x,gray_combined[y,x],amp[y,x],phase[y,x],edge[y,x],2]) # pattern3
####################################################################

    temp = np.array(data)
    df = pd.DataFrame(temp, columns =['y', 'x','gray_value','amp_4','phase_4', 'label'])
    df.to_csv('./step2/output'+filename+'/data'+str(i+1)+'.csv', index=False,)
    print('Create data ' +str(i+1))
toc()


# In[41]:


# output visualised pixels labelling shading, symbol and white background
savedir = './vistest/'
os.makedirs(savedir, exist_ok=True)

# Label check test#
img_shade = cv2.imread('./test/output/data68shading.png')
img_text = cv2.imread('./test/output/data68text.png')
img_combined = cv2.imread('./test/output/data68combined.png')
# get information per pixel 
gray_shade = cv2.cvtColor(img_shade,cv2.COLOR_RGB2GRAY)
gray_text = cv2.cvtColor(img_text,cv2.COLOR_RGB2GRAY)
gray_combined = cv2.cvtColor(img_combined,cv2.COLOR_RGB2GRAY)

# get image size
height, width = gray_combined.shape[:2]

# initialise
data =[]

### add FFT value as predictors ###
for y in range(height):
    for x in range(width):
        if gray_combined[y,x] == 255:
            data.append([0])
        elif (gray_combined[y,x] == 0) & (gray_combined[y,x] == gray_shade[y,x])            & (gray_combined[y,x] != gray_text[y,x]):
            data.append([1])
        elif (gray_combined[y,x] == 0) & (gray_combined[y,x] != gray_shade[y,x])            & (gray_combined[y,x] == gray_text[y,x]):
            data.append([2])
        else:
            data.append([1])
####################################

data = np.array(data).reshape([int(height),int(width)])
vis = copy.copy(data)
vis_3 = cv2.merge([vis,vis,vis])
count1 = 0
count2 = 0
for i in range(height):
    for j in range(width):
        if data[i,j] == 1:
            count1 += 1
            vis_3[i,j] = np.array([0,0,255])
        elif data[i,j] == 2:
            count2 += 1
            vis_3[i,j] = np.array([0,0,0])
        else:
            vis_3[i,j] = np.array([255,255,255])
# cv2.imwrite(savedir +'data1_sample.png', vis_3)
# cv2.imwrite(savedir +'data12_sample.png', vis_3)
print(count1,count2)


# In[42]:


### FFT update unit test###
# num = 6
# num_insert = int(num/2)
# img = cv2.imread('./test/output/data1combined.png',0)
# img_ex = addframe_update(num,255,img)
# # amp_4,phase_4 = FFT_update(img_ex, num)
# h = img_ex.shape[0] - (num_insert*2)+1
# w = img_ex.shape[1] - (num_insert*2)+1
# for  y in range(num_insert-1,img_ex.shape[0]-num_insert):
#     for x in range(num_insert-1,img_ex.shape[1]-num_insert):
#         img1 = img_ex[(y-num_insert+1):(y+num_insert+1),(x-num_insert+1):(x+num_insert+1)]
#         h1,w1 = img1.shape[0],img1.shape[1]
#         print(y,x,h1,w1)


# In[23]:


#### FFT unit test ####
# import itertools
# num = 5
# img = cv2.imread('../test/output/data68combined.png',0)
# img_ex = addframe(num,255,img)
# amp = []
# phase = []
# num_insert = 2 #int((num-1)/2)
# for  y in range(num_insert,img_ex.shape[0]-num_insert):
#     for x in range(num_insert,img_ex.shape[1]-num_insert):
#         img1 = img_ex[(y-num_insert):(y+num_insert+1),(x-num_insert):(x+num_insert+1)]
#         print(y,x,img1.shape[0],img1.hspae)
#         fimg = np.fft.fft2(img1)
#         #fshift = np.fft.fftshift(fimg)
#         a = np.abs(fimg)
#         p = np.angle(fimg)
# #         print(y,x)
# #         print(fimg)
# #         print(a)
# #         print(img1.shape)
#         for i,j in itertools.product(range(5),range(5)):
#             amp.append(a[i,j])
#             phase.append(p[i,j])
# #             amp.append(a[num_insert, num_insert])
# #             phase.append(p[num_insert, num_insert])
# # amp = np.reshape(amp,(img.shape[0],img.shape[1]))
# # amp.shape


# In[45]:


#### detect distance unittest ####
# num = 3
# img1 = cv2.imread('../test/output/data1combined.png',0)
# img2 = cv2.imread('../test/output/data1shading.png',0)
# img3 = cv2.imread('../test/output/data1text.png',0)
# img_ex = addframe(num,255,img1)
# img_ex_txt = addframe(num,255,img3)
# edge = detect_edge(img_ex,img2,img3,img_ex_txt)
# # amp = np.reshape(amp,(img.shape[0],img.shape[1]))
# edge = np.reshape(edge,(img1.shape[0],img.shape[1]))
# np.savetxt('./test_edge_detect.csv',edge, fmt ='%d', delimiter=',')


# In[ ]:




