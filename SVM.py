#!/usr/bin/env python
# coding: utf-8

# In[7]:


# import library
from glob import glob
import os
from os.path import join
import random
from sklearn import svm
from sklearn import datasets
import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import openpyxl
import cv2
import copy


# In[8]:


def split(split_ratio, img_src):
    # get file list
    files = glob(join(img_src, '*.csv'))
    # split data into train and test sets(8:2)
    train_file, test_file = train_test_split(files, train_size=split_ratio)
    # Extract file name
    for f in range(len(train_file)):
        train_file[f] = os.path.split(train_file[f])[1]
    for f in range(len(test_file)):
        test_file[f] = os.path.split(test_file[f])[1]
    return train_file, test_file


# In[9]:


# check file list
ck_train = join('./','./train_file_list.csv')
ck_test = join('./','./test_file_list.csv')
if (os.path.exists(ck_train)) and (os.path.exists(ck_test)):
    print('already exits files')
else:
    os.remove(ck_train)
    os.remove(ck_test)
    img_src = './step2/output31_csv' #select name folder after creating 200 data to separate training and test data
    train_file, test_file = split(0.8,img_src)
    df1 = pd.DataFrame(train_file)
    df2 = pd.DataFrame(test_file)
    # Save CSVfiles
    df1.to_csv('./train_file_list.csv', index=None)
    df2.to_csv('./test_file_list.csv', index=None)


# In[ ]:


# foldername: output5_csv, output21_csv, output31_csv, output6_3_csv, output32_16_csv, output5_de_csv 
foldername = 'output31_csv/'
Path = './step2/'+ foldername
# get file list
test_file = np.loadtxt('./test_file_list.csv', delimiter=',',skiprows=1, dtype='object')
train_file = np.loadtxt('./train_file_list.csv', delimiter=',',skiprows=1, dtype='object' )

# #### apply Support Vector Machine #### parameters:default

# clf = svm.SVC() # initialisation
# for num in range(len(train_file)):
#     df = pd.read_csv(Path+train_file[num])
# #### for adding labels of pixels on the edge ####
# #     train_X1,train_X2,train_y = df.iloc[:,:-2],df.iloc[:,-2],df.iloc[:,-1].values # X1,X2:dataframe y:array
# #     train_X2 = pd.get_dummies(train_X2,sparse=True) # one-hot encoding regarding edge label
# #     train_X = train_X1.join(train_X2).values # X:array
# #################################################
#     train_X,train_y = df.iloc[:,:-1].values,df.iloc[:,-1].values # before adding edge label
#     clf.fit(train_X,train_y)
#     print('Fitting '+str(num+1)+ '/' + str(len(train_file))) # learn 160 training sets

# # output model
# print('Save model')
# modelname: ./train5.learn, ./train21.learn, ./train31.learn, ./train6_3.learn,./train32_16.learn, ./train5_de.learn 
modelname = './train31.learn'
# joblib.dump(clf, modelname)

#### Create classification report and visualised images ####
# savedir: ./pre_img5X5, ./pre_img21X21, ./pre_img31X31, ./pre_img6X6_3, ./pre_img32X32_16, ./pre_img5X5_de
savedir = './pre_img31X31'
os.makedirs(savedir+'/SVC_vis',exist_ok=True) # create directory for predicted imgs 
os.makedirs(savedir+'/SVC',exist_ok=True) # create directory for unshaded txt based on predicted label
os.makedirs(savedir+'/shading',exist_ok=True) # create directory for shading txt

# load model
print('Load model')
clf4 = joblib.load(modelname)
# load test data
#savename: ./evaluation5.xlsx, ./evaluation21.xlsx, ./evaluation31.xlsx, ./evaluation6_3.xlsx, ./evaluation32_16.xlsx,
#          ./evaluation5_de.xlsx
savename = join('./','./evaluation31.xlsx')
if os.path.exists(savename):
    os.remove(savename)

### implement test file ###
for num in range(len(test_file)):
    df = pd.read_csv(Path + test_file[num])
    height,width = df.iloc[-1,0],df.iloc[-1,1]
#### for adding labels of pixels on the edge ####
#     test_X1,test_X2,test_y = df.iloc[:,:-2],df.iloc[:,-2],df.iloc[:,-1].values # X1,X2:dataframe y:array
#     test_X2 = pd.get_dummies(test_X2,sparse=True)
#     test_X = test_X1.join(test_X2).values # X:array
#################################################
    test_X,test_y = df.iloc[:,:-1].values,df.iloc[:,-1].values #before introducing edge label
    # result
    predicted_y = clf4.predict(test_X)
    print('Output ' +str(test_file[num]) +'['+ str(num+1) + '/' + str(len(test_file))+']')
    # evaluation(accuracy)
    report = classification_report(test_y, predicted_y, output_dict=True)
    sheetname = os.path.splitext(os.path.basename(test_file[num]))[0]
    result = pd.DataFrame(report).transpose()
    if os.path.exists(savename):
        with pd.ExcelWriter(savename, engine="openpyxl", mode="a") as writer:
            result.to_excel(writer, sheet_name=sheetname)
    else:
        with pd.ExcelWriter(savename, engine="openpyxl") as writer:
            result.to_excel(writer, sheet_name=sheetname)
### initialisation ###
    predicted_y = predicted_y.reshape([int(height+1),int(width+1)])
    test_y = test_y.reshape([int(height+1),int(width+1)])
#     train_y = train_y.reshape([int(height+1),int(width+1)]) # for sending predicted shading texts to a partner
    pre_vis =  copy.copy(predicted_y)
    pre_vis_3 = cv2.merge([pre_vis,pre_vis,pre_vis])
    pre_label = np.zeros_like(predicted_y, dtype=np.uint8)
    pre_shading_label = np.zeros_like(predicted_y, dtype=np.uint8)
    
### modifying label to restore the symbol texts ###
## output predicted image (Red:part of shading, Black: part of text)
# Note: the format is BGR when using Open CV
### restore unshaded txt and visualised images###
    for i in range(int(height)):
        for j in range(int(width)):
#             if predicted_y[i,j] == 1 and train_y[i,j] ==1:
            if predicted_y[i,j] == 1 and test_y[i,j] ==1:
                pre_label[i,j] = 255 #1
                pre_vis_3[i,j] = np.array([0,0,255]) # Color code #FF0000(Red)
#             elif predicted_y[i,j] == 1 and train_y[i,j] ==2:
            elif predicted_y[i,j] == 1 and test_y[i,j] ==2:
                pre_label[i,j] = 255
                pre_vis_3[i,j] = np.array([255,255,0]) # Color code #00FFFF(Cyan)
#             elif predicted_y[i,j] == 2 and train_y[i,j]==2:
            elif predicted_y[i,j] == 2 and test_y[i,j]==2:
                pre_label[i,j] = 0 #1
                pre_vis_3[i,j] = np.array([0,0,0]) # Color code #000000(Black)
#             elif predicted_y[i,j] ==2 and train_y[i,j] == 1:
            elif predicted_y[i,j] ==2 and test_y[i,j] == 1:
                pre_label[i,j] = 0 #1
                pre_vis_3[i,j] = np.array([0,140,255]) # Color code #FF8C00(DarkOrange)                
            else:
                pre_label[i,j] = 255 #1
                pre_vis_3[i,j] = np.array([255,255,255]) # Color code #FFFFFF(White)
####################################################

### modifying label to restore the only shading texts ###
    for k in range(int(height)):
        for l in range(int(width)):
            if predicted_y[k,l] != 1:
                pre_shading_label[k,l] = 255
            else:
                pre_shading_label[k,l] = 0
#########################################################
    pre_label = pre_label.astype(int)# change type (float -> int) 
    pre_shading_label = pre_shading_label.astype(int)# change type (float -> int)
    
### output img SVC / extract only shading ### 
#     print(pre_vis_3) # for check output
    cv2.imwrite(savedir+'/SVC/'+sheetname + 'SVC.png',pre_label) #output img using SVC
    cv2.imwrite(savedir+'/SVC_vis/'+sheetname + 'SVC_vis.png',pre_vis_3) #output img using SVC visualisation
    cv2.imwrite(savedir+'/shading/' +sheetname +'shading.png', pre_shading_label) #output shading img
    print('Now create img file'+'['+str(num+1)+'/'+str(len(test_file))+']') # testfile comment
#     print('Now create img file'+'['+str(num+1)+'/'+str(len(train_file))+']') # trainfile comment    
#############################################

# ## implement train file for checking overfitting ##
# for num in range(len(train_file)):
#     df = pd.read_csv(Path + train_file[num])
#     train_X,train_y = df.iloc[:,:-1].values,df.iloc[:,-1].values
#     # result
#     predicted_y = clf4.predict(train_X)
#     print('Output ' +str(train_file[num]) +'['+ str(num+1) + '/' + str(len(train_file))+']')
#     # evaluation(accuracy)
#     report = classification_report(train_y, predicted_y, output_dict=True)
#     sheetname = os.path.splitext(os.path.basename(train_file[num]))[0]
#     result = pd.DataFrame(report).transpose()
#     if os.path.exists(savename):
#         with pd.ExcelWriter(savename, engine="openpyxl", mode="a") as writer:
#             result.to_excel(writer, sheet_name=sheetname)
#     else:
#         with pd.ExcelWriter(savename, engine="openpyxl") as writer:
#             result.to_excel(writer, sheet_name=sheetname)
#####################################################

print('Complete task')  


# In[ ]:




