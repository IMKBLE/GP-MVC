import time
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import torch
import numpy as np
import scipy.io as sio
from sklearn.cluster import KMeans
import models.metrics as metrics
import matplotlib.pyplot as plt 
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

data_0 = sio.loadmat('rand1/label.mat')
data_dict=dict(data_0)
data0 = data_dict['label']
label_true = np.zeros((len(train_dataset_a)))
for i in range(len(train_dataset_a)):
    label_true[i]=data0[i]
label_true_all = np.zeros(len(data0))
for i in range(len(data0)):
    label_true_all[i]=data0[i]
#label_true_UN = np.zeros(len(untrain_dataset_a)*3/2)
#for i in range(2*len(untrain_dataset_a)):
#    label_true_UN[i]=data0[i+len(train_dataset_a)]
print(len(dataset_paired))
print(len(dataset_unpaired))
print(len(train_dataset_b))
#print(len(untrain_dataset_a))
n_clusters = 10
n_com = 100
dim1 = 216
dim2 = 76
dim3 = 64
# Create Model
model = create_model(opt)
visualizer = Visualizer(opt)
torch.cuda.synchronize()
start=time.time()
# Start Training
print('Start training')

#################################################
# Step1: Autoencoder
#################################################
print('step 1')
pre_epoch_AE = 15 # number of iteration for autoencoder pre-training
total_steps = 0
ACC_all=[]
NMI_all=[]
loss_ae = []
for epoch in range(1, pre_epoch_AE+1):
#    for i,(images_a, image_b) in enumerate(dataset_paired):

    for i in range(len(train_dataset_a)):

        images_a = torch.from_numpy(train_dataset_a[i]).view(1,1,dim1)
        images_b = torch.from_numpy(train_dataset_b[i]).view(1,1,dim2)
        images_c = torch.from_numpy(train_dataset_c[i]).view(1,1,dim3)
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - len(train_dataset_a) * (epoch - 1)
        model.set_input(images_a, images_b, images_c)
        model.optimize_parameters_pretrain_AE()
        loss_ae.append(model.loss_AE_pre.data.cpu())
        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors_AE_pre()
            visualizer.print_current_errors(epoch, epoch_iter, errors, iter_start_time)
#            if opt.display_id > 0:
#                visualizer.plot_current_errors(epoch, float(epoch_iter)/len(train_dataset_a), opt, errors)
    print('pretrain Autoencoder model (epoch %d, total_steps %d)' %
          (epoch, pre_epoch_AE)) 
    commonZ = []
    if epoch > 0:
        for i in range(len(train_dataset_a)):
            tempimage_a = torch.from_numpy(train_dataset_a[i]).view(1,1,dim1)
            tempimage_b = torch.from_numpy(train_dataset_b[i]).view(1,1,dim2) 
            tempimage_c = torch.from_numpy(train_dataset_c[i]).view(1,1,dim3)  
            model.set_input(tempimage_a, tempimage_b, tempimage_c)
            t_200 =np.array(model.test_commonZ().data.view(n_com).tolist())
            commonZ.append(t_200)
        ##kmeans result
        estimator = KMeans(n_clusters)
        estimator.fit(commonZ)
#        Z_path = 'step1Z'+ str(epoch)
#        sio.savemat(Z_path+'.mat',{'Z':commonZ})
        centroids =estimator.cluster_centers_
        label_pred = estimator.labels_
        acc = metrics.acc(label_true, label_pred)
        nmi = metrics.nmi(label_true, label_pred)
        ACC_all.append(acc)
        NMI_all.append(nmi)
        print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
                   % (acc, nmi))
        
centroids0 =estimator.cluster_centers_
########
center0 = torch.FloatTensor(centroids0).cuda()
model.clu.weights.data = center0
#########



fa_500 = []
fb_500 = []
fc_500 = []
B_undata1 = []
B_undata2 = []
fff = tempimage_a
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
sio.savemat('fakea1.mat',{'fa':fa_500})
sio.savemat('fakeb1.mat',{'fb':fb_500})
sio.savemat('fakec1.mat',{'fc':fc_500})

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
ACC_all.append(acc)
NMI_all.append(nmi)
print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
      % (acc, nmi))
#sio.savemat('ZAE.mat', {'commonZ':commonZ_step1})  

sio.savemat('commonZAE.mat', {'Z':commonZ_step1})  
#################################################
# Step2: CycleGAN
#################################################
test_dataset_A = np.array(list(train_dataset_a) + list(untrain_dataset_a))
test_dataset_B = np.array(list(train_dataset_b) + list(untrain_dataset_b)) 
test_dataset_C = np.array(list(train_dataset_c) + list(untrain_dataset_c)) 
loss_ae_g = []
loss_g_g = []
loss_da_g = []
loss_db_g = []

print('step 2')
pre_epoch_cycle = 5
# number of iteration for CycleGAN training
total_steps = 0
pair_unnum1 = len(train_dataset_a)+un_num
pair_unnum2 = len(train_dataset_a)+un_num*2
pair_unnum3 = len(train_dataset_a)+un_num*3
for epoch in range(1, pre_epoch_cycle+1):
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
            model.optimize_parameters_pretrain_cycleGAN()  
#            t_200 =np.array(model.test_commonZ().data.view(n_com).tolist())
#            commonZ_step22.append(t_200)
        elif i<pair_unnum1:
            j =np.random.randint(0,len(test_dataset_A),1)
            j = j[0]           
            images_a_2 = torch.from_numpy(test_dataset_A[j]).view(1,1,dim1)
            images_b_2 = torch.from_numpy(test_dataset_B[i]).view(1,1,dim2)
            images_c_2 = torch.from_numpy(test_dataset_C[i]).view(1,1,dim3)  
            model.set_input(images_a_2, images_b_2, images_c_2)
            model.optimize_parameters_pretrain_cycleGAN()  
#            com12, com13, com23 = model.test_2commonZ()
#            t_200 =np.array(com23.data.view(n_com).tolist())
#            commonZ_step22.append(t_200)
        elif i<pair_unnum2:
            j =np.random.randint(0,len(test_dataset_A),1)
            j = j[0]           
            images_a_2 = torch.from_numpy(test_dataset_A[i-un_num]).view(1,1,dim1)
            images_b_2 = torch.from_numpy(test_dataset_B[j]).view(1,1,dim2)
            images_c_2 = torch.from_numpy(test_dataset_C[i]).view(1,1,dim3)              
            model.set_input(images_a_2, images_b_2, images_c_2)
            model.optimize_parameters_pretrain_cycleGAN()
#            com12, com13, com23 = model.test_2commonZ()
#            t_200 =np.array(com13.data.view(n_com).tolist())
#            commonZ_step22.append(t_200)
        else:
            j =np.random.randint(0,len(test_dataset_A),1)
            j = j[0]           
            images_a_2 = torch.from_numpy(test_dataset_A[i-un_num]).view(1,1,dim1)
            images_b_2 = torch.from_numpy(test_dataset_B[i-un_num]).view(1,1,dim2)
            images_c_2 = torch.from_numpy(test_dataset_C[j]).view(1,1,dim3)              
            model.set_input(images_a_2, images_b_2, images_c_2)
            model.optimize_parameters_pretrain_cycleGAN()
#            com12, com13, com23 = model.test_2commonZ()
#            t_200 =np.array(com12.data.view(n_com).tolist())
#            commonZ_step22.append(t_200)
            

        loss_ae_g.append(model.loss_AE.data.cpu())
        loss_g_g.append(model.loss_GABC.data.cpu())
        loss_da_g.append(model.loss_D_A.data.cpu())
        loss_db_g.append(model.loss_D_B.data.cpu())
        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors_cycle()
            visualizer.print_current_errors(epoch, epoch_iter, errors, iter_start_time)
#            if opt.display_id > 0:
#                visualizer.plot_current_errors(epoch, float(epoch_iter)/unpaired_dataset_size, opt, errors)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, pre_epoch_cycle, time.time() - epoch_start_time))
    
#    estimator = KMeans(n_clusters)
#    estimator.fit(commonZ_step22)
#    centroids_step2 =estimator.cluster_centers_
#    label_pred = estimator.labels_
#    acc = metrics.acc(label_true_all, label_pred)
#    nmi = metrics.nmi(label_true_all, label_pred)
#    ACC_all.append(acc)
#    NMI_all.append(nmi)
#    print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
#             % (acc, nmi))    
    
    fa_500 = []
    fb_500 = []
    fc_500 = []
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
    sio.savemat('fakea2.mat',{'fa':fa_500})
    sio.savemat('fakeb2.mat',{'fb':fb_500})
    sio.savemat('fakec2.mat',{'fc':fc_500})


    commonZ_step2 = []             
    for i in range(len(test_dataset_A2000)):
        tempimage_a = torch.from_numpy(test_dataset_A2000[i]).view(1,1,dim1)
        tempimage_b = torch.from_numpy(test_dataset_B2000[i]).view(1,1,dim2)      
        tempimage_c = torch.from_numpy(test_dataset_C2000[i]).view(1,1,dim3)  
        model.set_input(tempimage_a, tempimage_b, tempimage_c)
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
    
ACC_all.append(acc)
NMI_all.append(nmi)
   
sio.savemat('commonZG.mat',{'Z':commonZ_step2})

q1 = 1.0 / (1.0 + (torch.sum(torch.pow(torch.unsqueeze(torch.FloatTensor(commonZ_step1), 1)-torch.FloatTensor(centroids0), 2), 2) ))
#q1 = 1.0 / (1.0 + (torch.sum(torch.pow(torch.unsqueeze(torch.FloatTensor(commonZ_step1), 1)-torch.FloatTensor(centroids0), 2), 2) ))
q = torch.t(torch.t(q1) / torch.sum(q1, 1))
p1 = torch.pow(q,2)/torch.sum(q,0)
p = torch.t(torch.t(p1)/torch.sum(p1,1))
#center = torch.FloatTensor(centroids).cuda()
#center = torch.FloatTensor(centroids_step2).cuda()
#model.clu.weights.data = center

#################################################
# Step3:  VIGAN
#################################################
print('step 3')
total_steps = 0
#eee = []
#ACC_all=[]
#NMI_all=[]
loss_ave = []
loss_temp = torch.zeros(1)
for epoch in range(1, opt.niter + opt.niter_decay + 1):
    if epoch>15:
        break
    epoch_start_time = time.time()
	# You can use paired and unpaired data to train. Here we only use paired samples to train.
    #for i,(images_a, images_b) in enumerate(dataset_paired):
    q = []    
    for i in range(len(test_dataset_A2000)):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - paired_dataset_size * (epoch - 1)

        images_a = torch.from_numpy(test_dataset_A2000[i]).view(1,1,dim1)
        images_b = torch.from_numpy(test_dataset_B2000[i]).view(1,1,dim2)
        images_c = torch.from_numpy(test_dataset_C2000[i]).view(1,1,dim3)
        pp_i = p[i].cuda()
        model.set_input_train(images_a,images_b,images_c,pp_i)
        model.optimize_AECL()
        q_i = model.q.data
        qi = q_i.view(n_clusters ).tolist()
        q.append(qi)
        loss_temp = model.loss_AE_CL.data.cpu()


    if total_steps % opt.print_freq == 0:
        errors = model.get_current_errors_AE_CL()
        visualizer.print_current_errors(epoch, epoch_iter, errors, iter_start_time)
        loss_ave.append((loss_temp).tolist())
        loss_temp = torch.zeros(1)
#        if opt.display_id > 0:
#            visualizer.plot_current_errors(epoch, float(epoch_iter)/paired_dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')
    

    fa_500 = []
    fb_500 = []
    fc_500 = []
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
    sio.savemat('fakea2.mat',{'fa':fa_500})
    sio.savemat('fakeb2.mat',{'fb':fb_500})
    sio.savemat('fakec2.mat',{'fc':fc_500})


#    commonZ_step2 = []             
#    for i in range(len(test_dataset_A2000)):
#        tempimage_a = torch.from_numpy(test_dataset_A2000[i]).view(1,1,dim1)
#        tempimage_b = torch.from_numpy(test_dataset_B2000[i]).view(1,1,dim2)      
#        tempimage_c = torch.from_numpy(test_dataset_C2000[i]).view(1,1,dim3)  
#        model.set_input(tempimage_a, tempimage_b, tempimage_c)
#        t_200 =np.array(model.test_commonZ().data.view(n_com).tolist())
#        commonZ_step2.append(t_200)
#    estimator = KMeans(n_clusters)
#    estimator.fit(commonZ_step2)
#    centroids_step2 =estimator.cluster_centers_
#    label_pred = estimator.labels_
#    acc = metrics.acc(label_true_all, label_pred)
#    nmi = metrics.nmi(label_true_all, label_pred)
#    ACC_all.append(acc)
#    NMI_all.append(nmi)
#    print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
#      % (acc, nmi))
#    
#ACC_all.append(acc)
#NMI_all.append(nmi)

    ##kmeans result
    commonZ = []
    for i in range(len(test_dataset_A2000)):
        tempimage_a = torch.from_numpy(test_dataset_A2000[i]).view(1,1,dim1)
        tempimage_b = torch.from_numpy(test_dataset_B2000[i]).view(1,1,dim2)  
        tempimage_c = torch.from_numpy(test_dataset_C2000[i]).view(1,1,dim3) 
        model.set_input(tempimage_a, tempimage_b,tempimage_c)
        t_200 =np.array(model.test_commonZ().data.view(n_com).tolist())
        commonZ.append(t_200)
    ##kmeans result
    estimator = KMeans(n_clusters)
    estimator.fit(commonZ)
    centroids =estimator.cluster_centers_
    label_pred = estimator.labels_
    acc = metrics.acc(label_true_all, label_pred)
    nmi = metrics.nmi(label_true_all, label_pred)
    ACC_all.append(acc)
    NMI_all.append(nmi)
    sio.savemat('acc.mat', {'ACC_all':ACC_all})
    sio.savemat('nmi.mat', {'NMI_all':NMI_all})
    sio.savemat('loss.mat', {'loss_all':loss_ave})
    Z_path = 'commonZ'+ str(epoch)
#    sio.savemat(Z_path+'.mat',{'Z':commonZ})
    print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
                   % (acc, nmi))

#    sio.savemat('commonZ.mat', {'commonZ':commonZ})
    sio.savemat(Z_path+'.mat',{'commonZ':commonZ})
    
#    loss_ave.append((loss_temp/len(train_dataset_a)).tolist())
    q = torch.FloatTensor(q)
    p1 = torch.pow(q,2)/torch.sum(q,0)
    p = torch.t(torch.t(p1)/torch.sum(p1,1))
    
    
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        #Z_path = 'commonZ'+ str(epoch)
        #sio.savemat(Z_path+'.mat',{'Z':commonZ})
        
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        model.update_learning_rate()
#sio.savemat('error.mat', {'error':eee})
ACC_all.append(acc)
NMI_all.append(nmi)
x=torch.linspace(1, len(ACC_all), steps=len(ACC_all))
x = x.numpy()
y_acc = torch.FloatTensor(ACC_all).numpy()
y_nmi = torch.FloatTensor(NMI_all).numpy()
y_loss = torch.FloatTensor(loss_ave).numpy()
plt.cla()
plt.plot(x, y_nmi, c='red', label='nmi')
plt.plot(x, y_acc, c='blue', label='acc')
#plt.plot(x, y_loss, c='green', label='loss')


sio.savemat('databaseA.mat',{'dataA':test_dataset_A2000})
sio.savemat('databaseB.mat',{'dataB':test_dataset_B2000})


  
print("Time used")
torch.cuda.synchronize()
end=time.time()
time = end - start
print(time)
