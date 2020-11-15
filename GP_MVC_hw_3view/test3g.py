import time
import os
import scipy.io as sio
from options.test_options import TestOptions
opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import util.util as util
import torch
import scipy.io as sio
from sklearn.cluster import KMeans
import models.metrics as metrics
import matplotlib.pyplot as plt 
opt.nThreads = 1   # test code only supports nThreads=1
opt.batchSize = 1  #test code only supports batchSize=1
opt.serial_batches = True # no shuffle

#load data
# Load data
data_loader = CreateDataLoader(opt)
dataset_paired, paired_dataset_size = data_loader.load_data_pair()
dataset_unpaired, unpaired_dataset_size = data_loader.load_data_unpair()
train_dataset_a = dataset_paired.dataset.train_data_a
train_dataset_b = dataset_paired.dataset.train_data_b
train_dataset_c = dataset_paired.dataset.train_data_c
untrain_dataset_a = dataset_unpaired.dataset.train_data_a
untrain_dataset_b = dataset_unpaired.dataset.train_data_b
untrain_dataset_c = dataset_unpaired.dataset.train_data_c
#sio.savemat('ua.mat',{'upaireda':untrain_dataset_a})
#sio.savemat('ub.mat',{'upairedb':untrain_dataset_b})
data_0 = sio.loadmat('rand10/label.mat')
data_dict=dict(data_0)
data0 = data_dict['label']
label_true = np.zeros((len(train_dataset_a)))
for i in range(len(train_dataset_a)):
    label_true[i]=data0[i]
label_true_all = np.zeros(len(data0))
for i in range(len(data0)):
    label_true_all[i]=data0[i]
    
model = create_model(opt)
visualizer = Visualizer(opt)
n_clusters = 10
n_com = 100
dim1 = 216
dim2 = 76
dim3 = 64
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

fa_500 = []
fb_500 = []
fc_500 = []
B_undata1 = []
B_undata2 = []
fff = torch.from_numpy(train_dataset_a[1]).view(1,1,dim1)
un_num = int((len(untrain_dataset_a))/2)
for i in range(un_num):
    tempimage_a1 = torch.from_numpy(untrain_dataset_a[i]).view(1,1,dim1)
    tempimage_b1 = torch.from_numpy(untrain_dataset_b[i]).view(1,1,dim2)
    tempimage_c1 = torch.from_numpy(untrain_dataset_c[i]).view(1,1,dim3)
    
    tempimage_a2 = torch.from_numpy(untrain_dataset_a[i + un_num]).view(1,1,dim1)
    tempimage_b2 = torch.from_numpy(untrain_dataset_b[i + un_num]).view(1,1,dim2)
    tempimage_c2 = torch.from_numpy(untrain_dataset_c[i + un_num]).view(1,1,dim3)    
    b_undata1 = tempimage_b1.view(1,dim2).tolist()
    b_undata2 = tempimage_b2.view(1,dim2).tolist()
    B_undata1.append(b_undata1)
    B_undata2.append(b_undata2)
    
      
    model.set_input(fff, tempimage_b1, tempimage_c1)
    dataset_fakeA = model.test_unpaired_a()
    data_fakeA = dataset_fakeA.data.view(1,dim1).tolist()
    
    model.set_input(tempimage_a1, fff, tempimage_c2)
    dataset_fakeB = model.test_unpaired_b()    
    data_fakeB = dataset_fakeB.data.view(1,dim2).tolist()
    
    model.set_input(tempimage_a2, tempimage_b2, fff)
    dataset_fakeC = model.test_unpaired_c()    
    data_fakeC = dataset_fakeC.data.view(1,dim3).tolist()
    
    fa_500.append(data_fakeA)
    fb_500.append(data_fakeB)
    fc_500.append(data_fakeC)
    
test_dataset_A2000 = np.array(list(train_dataset_a)+ list(fa_500) + list(untrain_dataset_a) )
test_dataset_B2000 = np.array(list(train_dataset_b) + list(B_undata1) + list(fb_500) + list(B_undata2)) 
test_dataset_C2000 = np.array(list(train_dataset_c) + list(untrain_dataset_c) + list(fc_500)) 


commonZ_step1 = []
for i in range(len(test_dataset_A2000)):
    tempimage_a = torch.from_numpy(test_dataset_A2000[i]).view(1,1,dim1)
    tempimage_b = torch.from_numpy(test_dataset_B2000[i]).view(1,1,dim2) 
    tempimage_c = torch.from_numpy(test_dataset_C2000[i]).view(1,1,dim3)  
    model.set_input(tempimage_a, tempimage_b, tempimage_c)
    t_200 =np.array(model.test_commonZ().data.view(n_com).tolist())
    commonZ_step1.append(t_200)
estimator = KMeans(n_clusters)
estimator.fit(commonZ_step1)
centroids =estimator.cluster_centers_
label_pred = estimator.labels_
acc = metrics.acc(label_true_all, label_pred)
nmi = metrics.nmi(label_true_all, label_pred)

print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
      % (acc, nmi))


sio.savemat('commonZ.mat', {'commonZ':commonZ_step1})  

#################################################
# Step2: CycleGAN
#################################################
test_dataset_A = np.array(list(train_dataset_a) + list(untrain_dataset_a))
test_dataset_B = np.array(list(train_dataset_b) + list(untrain_dataset_b)) 
test_dataset_C = np.array(list(train_dataset_c) + list(untrain_dataset_c)) 


# number of iteration for CycleGAN training
total_steps = 0
pair_unnum1 = len(train_dataset_a)+un_num
pair_unnum2 = len(train_dataset_a)+un_num*2
pair_unnum3 = len(train_dataset_a)+un_num*3
opt.print_freq = 500
pre_epoch_cycle = 5
for epoch in range(1):
    commonZ_step22 = []
    epoch_start_time = time.time()
    #for i,(images_a, images_b) in enumerate(dataset_unpaired):
    for i in range(pair_unnum3):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - unpaired_dataset_size * (epoch - 1)
        if i<len(train_dataset_a):
            images_a_2 = torch.from_numpy(test_dataset_A[i]).view(1,1,dim1)
            images_b_2 = torch.from_numpy(test_dataset_B[i]).view(1,1,dim2)
            images_c_2 = torch.from_numpy(test_dataset_C[i]).view(1,1,dim3)
            model.set_input(images_a_2, images_b_2, images_c_2)
            t_200 =np.array(model.test_commonZ().data.view(n_com).tolist())
            commonZ_step22.append(t_200)
        elif i<pair_unnum1:
            j =np.random.randint(0,len(test_dataset_A),1)
            j = j[0]           
            images_a_2 = torch.from_numpy(test_dataset_A[j]).view(1,1,dim1)
            images_b_2 = torch.from_numpy(test_dataset_B[i]).view(1,1,dim2)
            images_c_2 = torch.from_numpy(test_dataset_C[i]).view(1,1,dim3)  
            model.set_input(images_a_2, images_b_2, images_c_2)
            com12, com13, com23 = model.test_2commonZ()
            t_200 =np.array(com23.data.view(n_com).tolist())
            commonZ_step22.append(t_200)
        elif i<pair_unnum2:
            j =np.random.randint(0,len(test_dataset_A),1)
            j = j[0]           
            images_a_2 = torch.from_numpy(test_dataset_A[i-un_num]).view(1,1,dim1)
            images_b_2 = torch.from_numpy(test_dataset_B[j]).view(1,1,dim2)
            images_c_2 = torch.from_numpy(test_dataset_C[i]).view(1,1,dim3)              
            model.set_input(images_a_2, images_b_2, images_c_2)
            com12, com13, com23 = model.test_2commonZ()
            t_200 =np.array(com13.data.view(n_com).tolist())
            commonZ_step22.append(t_200)
        else:
            j =np.random.randint(0,len(test_dataset_A),1)
            j = j[0]           
            images_a_2 = torch.from_numpy(test_dataset_A[i-un_num]).view(1,1,dim1)
            images_b_2 = torch.from_numpy(test_dataset_B[i-un_num]).view(1,1,dim2)
            images_c_2 = torch.from_numpy(test_dataset_C[j]).view(1,1,dim3)              
            model.set_input(images_a_2, images_b_2, images_c_2)
            com12, com13, com23 = model.test_2commonZ()
            t_200 =np.array(com12.data.view(n_com).tolist())
            commonZ_step22.append(t_200)
            
    
    estimator = KMeans(n_clusters)
    estimator.fit(commonZ_step22)
    centroids_step2 =estimator.cluster_centers_
    label_pred = estimator.labels_
    acc = metrics.acc(label_true_all, label_pred)
    nmi = metrics.nmi(label_true_all, label_pred)

    print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
             % (acc, nmi))    

















# test
#groundtruth = []
#rmse_x = []
#rmse_y = []
#fa_600 = []
#fb_600 = []
#print(len(untrain_dataset_a))
#print(len(test_dataset_a))
#for i in range(len(untrain_dataset_a)):
#        #groundtruth.append(lab_a)
#        #print("THis is the ith " , i)
#        #print(untrain_dataset_a)
#    images_a = torch.from_numpy(untrain_dataset_a[i]).view(1,1,1750)
#    images_b = torch.from_numpy(untrain_dataset_b[i]).view(1,1,79)
#    #groundtruth.append(lab_a)
#    model.set_input(images_a, images_b)
#    dataset_fakeA, dataset_fakeB = model.test_unpaired()
#    data_fakeA = dataset_fakeA.data.view(1,1750).tolist()
#    data_fakeB = dataset_fakeB.data.view(1,79).tolist()
#    fa_600.append(data_fakeA)
#    fb_600.append(data_fakeB)
#
#    #visuals = model.get_current_visuals()
#    #img_path = 'image'+ str(i)
#    #print('process image... %s' % img_path)
#    #visualizer.save_images(webpage, visuals, img_path)
#############################
#test_dataset_A2400 = np.array(list(test_dataset_a) + list(untrain_dataset_a) + list(fa_600) )
#test_dataset_B2400 = np.array(list(test_dataset_b) + list(fb_600) + list(untrain_dataset_b) )
#
##test_dataset_A2000 = np.array(list(train_dataset_a))
##test_dataset_B2000 = np.array(list(train_dataset_b))
#################################
##for i in range(len(test_dataset_A2000)):
##    print(test_dataset_A2000[i])
##    print(test_dataset_B2000[i])
#
#
##print(test_dataset_2000)
##print('test_dataset_2000')
#
##for i, (images_a, images_b) in enumerate(dataset_paired):
##    print(images_a)
##    print(torch.from_numpy(test_dataset_A2000[0]).view(1,1,28,28))
#commonZ = []
#for i in range(len(test_dataset_A2400)):
#    image_a = torch.from_numpy(test_dataset_A2400[i]).view(1,1,1750)
#    print(i)
#    image_b = torch.from_numpy(test_dataset_B2400[i]).view(1,1,79)
#        #groundtruth.append(lab_a)
#        #print("THis is the ith " , i)
#        #print(untrain_dataset_a)
#        
#    model.set_input(image_a, image_b)
#    t_200 =np.array(model.test_commonZ().data.tolist())
#    commonZ.append(t_200)
# 
##    visuals = model.get_current_visuals()
##    img_path = 'image'+ str(i)
##    print('process image... %s' % img_path)
##    visualizer.save_images(webpage, visuals, img_path)
#    
##print(commonZ)
####################################    
#sio.savemat('commonZ.mat',{'Z':commonZ})
#sio.savemat('databaseA.mat',{'dataA':test_dataset_A2400})
#sio.savemat('databaseB.mat',{'dataB':test_dataset_B2400})
#sio.savemat('fakea.mat',{'fa':fa_600})
#sio.savemat('fakeb.mat',{'fb':fb_600})
######################################
#
#
#
#
##sio.savemat('lab.mat',{'groundtruth':groundtruth})
##print(np.mean(rmse_y))
##print(np.mean(rmse_x))
##print((np.mean(rmse_x)+np.mean(rmse_y))/2)
##webpage.save()
