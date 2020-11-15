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
test_dataset_a = dataset_paired.dataset.test_data_a
test_dataset_b = dataset_paired.dataset.test_data_b
untrain_dataset_a = dataset_unpaired.dataset.train_data_a
untrain_dataset_b = dataset_unpaired.dataset.train_data_b
#sio.savemat('ua.mat',{'upaireda':untrain_dataset_a})
#sio.savemat('ub.mat',{'upairedb':untrain_dataset_b})
data_0 = sio.loadmat('rand10/label.mat')
data_dict=dict(data_0)
data0 = data_dict['label']
label_true = np.zeros((len(train_dataset_a)))
for i in range(len(train_dataset_a)):
    label_true[i]=data0[i]
label_true_all = np.zeros(len(train_dataset_a)+2*len(untrain_dataset_a))
for i in range(len(train_dataset_a)+2*len(untrain_dataset_a)):
    label_true_all[i]=data0[i]
label_true_UN = np.zeros(2*len(untrain_dataset_a))
for i in range(2*len(untrain_dataset_a)):
    label_true_UN[i]=data0[i+len(train_dataset_a)]
model = create_model(opt)
visualizer = Visualizer(opt)
n_clusters = 10
n_com = 100
dim1 = 4000
dim2 = 5000
dim3 = 5000
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

ACC_all=[]
NMI_all=[]
fa_500 = []
fb_500 = []
for i in range(len(untrain_dataset_a)):
    tempimage_a = torch.from_numpy(untrain_dataset_a[i]).view(1,1,dim1)
    tempimage_b = torch.from_numpy(untrain_dataset_b[i]).view(1,1,dim2)      
    model.set_input(tempimage_a, tempimage_b)
    dataset_fakeA, dataset_fakeB, t1_200, t2_200 = model.test_unpaired()
    data_fakeA = dataset_fakeA.data.view(1,dim1).tolist()
    data_fakeB = dataset_fakeB.data.view(1,dim2).tolist()
    fa_500.append(data_fakeA)
    fb_500.append(data_fakeB)
    
test_dataset_A2000 = np.array(list(train_dataset_a) + list(untrain_dataset_a) + list(fa_500))
test_dataset_B2000 = np.array(list(train_dataset_b) + list(fb_500) + list(untrain_dataset_b)) 
sio.savemat('fakea2.mat',{'fa':fa_500})
sio.savemat('fakeb2.mat',{'fb':fb_500})

commonZ_step2 = []             
for i in range(len(test_dataset_A2000)):
    tempimage_a = torch.from_numpy(test_dataset_A2000[i]).view(1,1,dim1)
    tempimage_b = torch.from_numpy(test_dataset_B2000[i]).view(1,1,dim2)      
    model.set_input(tempimage_a, tempimage_b)
    t_200 =np.array(model.test_commonZ().data.view(n_com).tolist())
    commonZ_step2.append(t_200)
estimator = KMeans(n_clusters)
estimator.fit(commonZ_step2)
centroids_step2 =estimator.cluster_centers_
label_pred = estimator.labels_
acc = metrics.acc(label_true_all, label_pred)
nmi = metrics.nmi(label_true_all, label_pred)
ACC_all.append(acc)
NMI_all.append(nmi)
print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
      % (acc, nmi))

sio.savemat('commonZ.mat', {'commonZ':commonZ_step2})





















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
