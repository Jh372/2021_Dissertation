#!/usr/bin/env python
# coding: utf-8

# In[14]:


#import library
import os
from pathlib import Path
from pdf2image import convert_from_path
import glob
import cv2
import numpy as np


# In[6]:


#### convert format from PDF into PNG ####
os.makedirs('./pdf_file',exist_ok=True) # create directory
num=300
for x in glob.glob('./pdf_file/*.pdf'):
    pdf_path = Path(x)
    
    #convert pdf to png
    pages = convert_from_path(str(pdf_path), dpi=num)
    
    #save image per page
    image_dir = Path('./image_file')
    for i, page in enumerate(pages):
        file_name = pdf_path.stem + '_{:02d}'.format(i + 1) + '.png'
        image_path = image_dir / file_name
        #save png
        page.save(str(image_path), 'PNG')
##########################################


# In[27]:


#### binarisation and Gaussian blur to crop shading pieces and text blocks using this image ####
os.makedirs('./test',exist_ok=True) # create directory for artificial images 
#read image 
#the final number of the image is 04,05: for cropping text blocks, 10,12: for cropping shading pieces
filename ='heft1pg01-15 2021-06-04 08_48_40_04.png'
# filename ='heft1pg01-15 2021-06-04 08_48_40_05.png'
# filename ='heft1pg01-15 2021-06-04 08_48_40_10.png'
# filename ='heft1pg01-15 2021-06-04 08_48_40_12.png'

img = cv2.imread('./image_file/'+filename,0)

# Otsu method w/o GaussianBlur
ret,th1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Otsu method with Gaussian Blur
blur = cv2.GaussianBlur(img,(5,5),0)
ret2,th2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#output
# cv2.imwrite('./test/sample2.png', th1)
cv2.imwrite('./test/text1.png',th2) # heft1pg01-15 2021-06-04 08_48_40_04.png
# cv2.imwrite('./test/text2.png',th2) # heft1pg01-15 2021-06-04 08_48_40_05.png
# cv2.imwrite('./test/shade1.png',th2) # heft1pg01-15 2021-06-04 08_48_40_10.png
# cv2.imwrite('./test/shade2.png',th2) # heft1pg01-15 2021-06-04 08_48_40_12.png
##############################################################################################


# In[24]:


#### superimpose shading blocks into text blocks to create artificial combined images ####
savedir = './test'
os.makedirs(savedir+'/output',exist_ok=True) # create directory for artificial images 

for i in range(20):
    test = cv2.imread('./test/crop/test'+str(i+1)+'.png')
    h_text,w_text =test.shape[:2]
    inv1 = cv2.bitwise_not(test)
    gray1 =cv2.cvtColor(inv1, cv2.COLOR_RGB2GRAY)
    for j in range(10):
        shade =  cv2.imread('./test/crop/shading'+str(j+1)+'.png')
        h_shade, w_shade = shade.shape[:2]
        if w_text%w_shade != 0:
            repeat_shade = (w_text/w_shade)+1
        else:
            repeat_shade = (w_text/w_shade)
        # join the same shading pieces together to fit the width of each symbol block
        shade_join = np.tile(shade,(1,int(repeat_shade),1))
        # crop the shading block into the same width of each symbol block
        shade_join = shade_join[0:h_text,0:w_text]

        cv2.imwrite('./test/output1/data'+str((i*10)+j+1)+'text.png',np.array(test))
        cv2.imwrite('./test/output1/data'+str((i*10)+j+1)+'shading.png',np.array(shade_join))
        inv2 = cv2.bitwise_not(shade_join)
        gray2 =cv2.cvtColor(inv2, cv2.COLOR_RGB2GRAY)
        result = cv2.add(gray1,gray2)
        result = cv2.bitwise_not(result)
        cv2.imwrite('./test/output1/data'+str((i*10)+j+1)+'combined.png',np.array(result))
########################################################################################


# In[ ]:




